"""Localização de caixas de remédio sobre mesa branca: OpenCV + Gemini.

Fluxo:
  1. A câmera é lida continuamente numa thread (sempre temos o frame mais recente).
  2. Quando a cena fica parada (ou ao apertar ESPAÇO) o frame é enviado ao Gemini.
  3. O Gemini devolve cada caixa: centro, ângulo, box_2d e o texto escrito nela.
  4. O OpenCV mede centro e ângulo de novo na região indicada pelo Gemini
     (watershed + minAreaRect) e valida/refina o resultado da IA.
  5. O resultado é impresso como JSON e salvo em saida/ultimo_resultado.json.

O Gemini e o OpenCV medem centro e ângulo de forma independente. Se concordam, usa-se
o valor do OpenCV (precisão de pixel); senão, o do Gemini. Os dois ficam no JSON.

Referencial da saída ("centro", "angulo_graus"): origem no CENTRO da câmera (imagem),
x -> direita, y -> cima, em pixels. angulo_graus = ângulo do lado MAIOR da caixa em
relação ao eixo x, em [-90, 90); positivo = anti-horário.

Teclas na janela:
  ESPAÇO  analisa o frame atual agora
  b       captura o fundo (mesa VAZIA) -> melhora muito a segmentação
  a       liga/desliga a análise automática quando a cena para
  s       salva o frame atual em saida/
  q/ESC   sai

Uso:
  python vision_remedios.py                 # câmera 0
  python vision_remedios.py --camera 1
  python vision_remedios.py --imagem foto.jpg   # analisa uma imagem e sai
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

PASTA_SAIDA = Path(__file__).parent / "saida"
ARQ_CALIBRACAO = Path(__file__).parent / "calibracao.json"

PROMPT = """Você é o sistema de visão de um braço robótico SCARA.
A imagem é uma vista de cima (câmera a ~85 cm) de uma mesa branca.
Detecte TODAS as caixas de remédio (embalagens de papelão) visíveis.

Regras:
- Uma entrada por caixa física. Não junte caixas encostadas; não divida uma caixa em duas.
- box_2d = [ymin, xmin, ymax, xmax] normalizado de 0 a 1000, justo nas bordas da caixa
  (inclua a caixa inteira, sem sombra e sem margem).
- centro = [y, x] normalizado de 0 a 1000: o ponto central da face de cima da caixa
  (onde as diagonais da caixa se cruzam). Seja o mais preciso possível: o robô vai pegar ali.
- eixo = [[y1, x1], [y2, x2]] normalizado de 0 a 1000: os dois pontos nas extremidades do
  EIXO MAIOR da caixa, ou seja, o ponto médio de cada um dos dois LADOS CURTOS da caixa.
  A linha entre eles atravessa a caixa no sentido do comprimento, passando pelo centro.
- Ignore sombras, reflexos, a mesa, o braço robótico, cabos e objetos que não sejam caixas de remédio.
- "nome": nome comercial principal escrito na caixa (ou "" se ilegível).
- "texto": todo o texto legível na face visível, na ordem de leitura, separado por " | ".
- "confianca": 0.0 a 1.0, sua confiança de que é uma caixa de remédio.
- Se não houver caixas, retorne lista vazia.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "caixas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "centro": {"type": "array", "items": {"type": "integer"}},
                    "eixo": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                    "nome": {"type": "string"},
                    "texto": {"type": "string"},
                    "confianca": {"type": "number"},
                },
                "required": ["box_2d", "centro", "eixo", "nome", "texto", "confianca"],
            },
        }
    },
    "required": ["caixas"],
}


# --------------------------------------------------------------------------- câmera

class Camera:
    """Lê a câmera numa thread para que o frame entregue seja sempre o mais recente."""

    def __init__(self, indice, largura, altura):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(indice, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {indice}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.rodando = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.rodando:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame

    def ler(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def fechar(self):
        self.rodando = False
        self.cap.release()


# --------------------------------------------------------------------------- OpenCV

class Segmentador:
    """Separa objetos da mesa branca. Com fundo capturado usa diferença de fundo;
    sem fundo, usa saturação + bordas (funciona, mas é menos robusto com caixas brancas)."""

    def __init__(self):
        self.fundo = None

    def capturar_fundo(self, frame):
        self.fundo = cv2.GaussianBlur(frame, (5, 5), 0)

    def mascara(self, frame):
        borrado = cv2.GaussianBlur(frame, (5, 5), 0)
        if self.fundo is not None and self.fundo.shape == frame.shape:
            lab_f = cv2.cvtColor(borrado, cv2.COLOR_BGR2LAB).astype(np.int16)
            lab_b = cv2.cvtColor(self.fundo, cv2.COLOR_BGR2LAB).astype(np.int16)
            diff = np.abs(lab_f - lab_b)
            # Luminância pesa menos para não pegar sombra suave como objeto.
            dist = (0.5 * diff[..., 0] + diff[..., 1] + diff[..., 2]).clip(0, 255).astype(np.uint8)
            _, m = cv2.threshold(dist, 18, 255, cv2.THRESH_BINARY)
        else:
            hsv = cv2.cvtColor(borrado, cv2.COLOR_BGR2HSV)
            m_cor = cv2.inRange(hsv, (0, 45, 0), (180, 255, 255))      # tudo que tem cor
            m_escuro = cv2.inRange(hsv, (0, 0, 0), (180, 255, 150))    # tudo que é escuro
            cinza = cv2.cvtColor(borrado, cv2.COLOR_BGR2GRAY)
            bordas = cv2.dilate(cv2.Canny(cinza, 40, 120), None, iterations=2)
            m = m_cor | m_escuro | bordas

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        # Preenche buracos (texto/desenho branco dentro da caixa).
        contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cheia = np.zeros_like(m)
        cv2.drawContours(cheia, contornos, -1, 255, cv2.FILLED)
        return cheia


def separar_objetos(frame, mascara, caixas_px, centros_px):
    """Watershed usando o centro de cada caixa (dado pelo Gemini) como semente.
    Separa caixas encostadas que na máscara viram uma mancha só.
    Devolve uma máscara (imagem inteira) por caixa, na mesma ordem."""
    marcadores = np.zeros(mascara.shape, np.int32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    marcadores[cv2.dilate(mascara, k, iterations=2) == 0] = 1  # fundo certo
    for i, ((x1, y1, x2, y2), centro) in enumerate(zip(caixas_px, centros_px)):
        centro = (int(centro[0]), int(centro[1]))
        eixos = (max(2, int((x2 - x1) * 0.15)), max(2, int((y2 - y1) * 0.15)))
        cv2.ellipse(marcadores, centro, eixos, 0, 0, 360, i + 2, cv2.FILLED)
    cv2.watershed(cv2.GaussianBlur(frame, (5, 5), 0), marcadores)
    return [((marcadores == i + 2).astype(np.uint8) * 255) & mascara for i in range(len(caixas_px))]


def refinar_caixa(regiao, caixa_px):
    """Procura, dentro da caixa do Gemini, o contorno do objeto e devolve um
    retângulo rotacionado (centro, tamanho, ângulo). None se não achar algo coerente."""
    x1, y1, x2, y2 = caixa_px
    h_img, w_img = regiao.shape
    pad = int(0.08 * max(x2 - x1, y2 - y1))
    ax1, ay1 = max(0, x1 - pad), max(0, y1 - pad)
    ax2, ay2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
    recorte = regiao[ay1:ay2, ax1:ax2]
    if recorte.size == 0:
        return None

    contornos, _ = cv2.findContours(recorte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None
    maior = max(contornos, key=cv2.contourArea)
    area_gemini = max(1, (x2 - x1) * (y2 - y1))
    razao = cv2.contourArea(maior) / area_gemini
    if not 0.45 <= razao <= 1.6:
        return None

    (cx, cy), (w, h), ang = cv2.minAreaRect(maior)
    cx, cy = cx + ax1, cy + ay1
    # Normaliza: largura = lado maior, ângulo do lado maior em [-90, 90).
    if w < h:
        w, h = h, w
        ang += 90
    ang = (ang + 90) % 180 - 90
    pontos = cv2.boxPoints(((cx, cy), (w, h), ang))
    return {"centro": (cx, cy), "tamanho": (w, h), "angulo": ang, "pontos": pontos}


# --------------------------------------------------------------------------- calibração

def carregar_homografia():
    """calibracao.json opcional: {"H": [[...],[...],[...]]} mapeando pixel -> mm
    no referencial do robô. Será gerado depois com o tabuleiro xadrez."""
    if ARQ_CALIBRACAO.exists():
        dados = json.loads(ARQ_CALIBRACAO.read_text(encoding="utf-8"))
        return np.array(dados["H"], dtype=np.float64)
    return None


def px_para_mm(H, x, y):
    p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), H)
    return float(p[0, 0, 0]), float(p[0, 0, 1])


# --------------------------------------------------------------------------- Gemini

# Usados em ordem quando o modelo principal está sobrecarregado/indisponível.
# gemini-robotics-er é o modelo do Google voltado para localização de objetos para robôs.
MODELOS_RESERVA = ["gemini-robotics-er-2-preview", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]


class AnalisadorGemini:
    def __init__(self, modelo):
        chave = os.getenv("GEMINI_API_KEY")
        if not chave:
            raise RuntimeError("Defina GEMINI_API_KEY no arquivo .env")
        # Timeout por chamada e poucas tentativas: com modelo sobrecarregado é melhor
        # pular logo para o próximo do que o SDK ficar repetindo por minutos.
        timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "90"))
        self.cliente = genai.Client(api_key=chave, http_options=types.HttpOptions(
            timeout=int(timeout_s * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        ))
        self.modelos = [modelo] + [m for m in MODELOS_RESERVA if m != modelo]
        self.modelo = modelo  # modelo que respondeu por último

    def detectar(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Falha ao codificar o frame")
        conteudo = [types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SCHEMA,
        )
        ultimo_erro = None
        for modelo in self.modelos:
            try:
                resp = self.cliente.models.generate_content(model=modelo, contents=conteudo, config=config)
            except errors.APIError as e:
                if e.code in (404, 429, 500, 503, 504):
                    print(f"[aviso] {modelo} indisponível ({e.code}), tentando o próximo...", flush=True)
                    ultimo_erro = e
                    continue
                raise
            except Exception as e:  # timeout / rede
                print(f"[aviso] {modelo} falhou ({type(e).__name__}), tentando o próximo...", flush=True)
                ultimo_erro = e
                continue
            self.modelo = modelo
            return json.loads(resp.text).get("caixas", [])
        raise RuntimeError(f"Nenhum modelo Gemini respondeu: {ultimo_erro}")


# Limites para considerar que Gemini e OpenCV estão falando da mesma caixa.
DIVERGENCIA_MAX_CENTRO = 0.25   # fração do lado menor da caixa
DIVERGENCIA_MAX_ANGULO = 15.0   # graus


def normalizar_angulo(a):
    return (a + 90) % 180 - 90


def diferenca_angulo(a, b):
    """Diferença entre orientações de caixa (simétricas a cada 180°), em [0, 90]."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def analisar(frame, gemini, segmentador, H, conf_minima):
    t0 = time.perf_counter()
    brutas = gemini.detectar(frame)
    t_gemini = time.perf_counter() - t0

    h_img, w_img = frame.shape[:2]
    ox, oy = w_img / 2, h_img / 2

    def relativo(x, y):
        """Pixel da imagem -> referencial da câmera (origem no centro, y para cima)."""
        return {"x": round(float(x - ox), 1), "y": round(float(oy - y), 1)}

    validas = []
    for det in brutas:
        b, c = det.get("box_2d", []), det.get("centro", [])
        if len(b) != 4 or det.get("confianca", 0) < conf_minima:
            continue
        ymin, xmin, ymax, xmax = b
        x1, x2 = sorted((int(xmin / 1000 * w_img), int(xmax / 1000 * w_img)))
        y1, y2 = sorted((int(ymin / 1000 * h_img), int(ymax / 1000 * h_img)))
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        if len(c) == 2:
            gx, gy = c[1] / 1000 * w_img, c[0] / 1000 * h_img
        else:
            gx, gy = (x1 + x2) / 2, (y1 + y2) / 2
        validas.append((det, (x1, y1, x2, y2), (gx, gy)))

    mascara = segmentador.mascara(frame)
    regioes = separar_objetos(frame, mascara, [v[1] for v in validas], [v[2] for v in validas])
    caixas = []
    for (det, (x1, y1, x2, y2), (gx, gy)), regiao in zip(validas, regioes):
        # Ângulo da IA a partir dos dois pontos do eixo maior (y para cima = sinal invertido).
        g_ang, eixo = None, det.get("eixo", [])
        if len(eixo) == 2 and all(len(p) == 2 for p in eixo):
            (ya, xa), (yb, xb) = eixo
            dx = (xb - xa) / 1000 * w_img
            dy = (ya - yb) / 1000 * h_img
            if np.hypot(dx, dy) > 5:
                g_ang = normalizar_angulo(float(np.degrees(np.arctan2(dy, dx))))
        gem = {"centro": relativo(gx, gy),
               "angulo_graus": None if g_ang is None else round(g_ang, 1),
               "eixo": [relativo(p[1] / 1000 * w_img, p[0] / 1000 * h_img) for p in eixo] if g_ang is not None else None,
               "box_2d": det["box_2d"]}

        ref = refinar_caixa(regiao, (x1, y1, x2, y2))
        ocv, diverg, concordam = None, None, False
        if ref:
            # refinar_caixa mede na convenção da imagem (y para baixo): inverte o sinal.
            o_ang = normalizar_angulo(-ref["angulo"])
            ocx, ocy = ref["centro"]
            ocv = {"centro": relativo(ocx, ocy), "angulo_graus": round(o_ang, 2)}
            d_centro = float(np.hypot(ocx - gx, ocy - gy))
            d_ang = None if g_ang is None else diferenca_angulo(o_ang, g_ang)
            diverg = {"centro_px": round(d_centro, 1),
                      "angulo_graus": None if d_ang is None else round(d_ang, 1)}
            # O centro decide se é a mesma caixa. O ângulo divergente só gera alerta:
            # a medida geométrica do OpenCV é mais confiável que a estimativa da IA.
            concordam = d_centro <= DIVERGENCIA_MAX_CENTRO * min(ref["tamanho"])
            diverg["alerta_angulo"] = d_ang is not None and d_ang > DIVERGENCIA_MAX_ANGULO

        if concordam:
            # Mesma caixa segundo os dois: usa o OpenCV, que tem precisão de pixel.
            cx, cy = ref["centro"]
            angulo, fonte = ocv["angulo_graus"], "gemini+opencv"
            w, h = ref["tamanho"]
            pontos = ref["pontos"]
        else:
            cx, cy, angulo = gx, gy, gem["angulo_graus"]
            fonte = "gemini (opencv divergiu)" if ref else "gemini"
            w, h = max(x2 - x1, y2 - y1), min(x2 - x1, y2 - y1)
            if angulo is not None:
                pontos = cv2.boxPoints(((cx, cy), (w, h), -angulo))
            else:
                pontos = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        centro = relativo(cx, cy)
        item = {
            "id": len(caixas) + 1,
            "nome": det.get("nome", ""),
            "texto": det.get("texto", ""),
            "confianca": round(float(det.get("confianca", 0)), 3),
            "centro": centro,
            "angulo_graus": angulo,
            "distancia_centro_px": round(float(np.hypot(centro["x"], centro["y"])), 1),
            "direcao_graus": round(float(np.degrees(np.arctan2(centro["y"], centro["x"]))), 1),
            "fonte": fonte,
            "gemini": gem,
            "opencv": ocv,
            "divergencia": diverg,
            "tamanho_px": [round(float(w), 1), round(float(h), 1)],
            "centro_imagem_px": [round(float(cx), 1), round(float(cy), 1)],
            "cantos_imagem_px": [[round(float(x), 1), round(float(y), 1)] for x, y in pontos],
        }
        if H is not None:
            item["centro_mm"] = [round(v, 2) for v in px_para_mm(H, cx, cy)]
        caixas.append(item)

    return {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "modelo": gemini.modelo,
        "resolucao": [w_img, h_img],
        "referencial": "origem no centro da câmera; x -> direita, y -> cima (px); "
                       "angulo = lado maior da caixa vs eixo x, positivo = anti-horário",
        "tempo_gemini_s": round(t_gemini, 2),
        "calibrado_mm": H is not None,
        "total": len(caixas),
        "caixas": caixas,
    }


# --------------------------------------------------------------------------- visualização

def desenhar(frame, resultado):
    vis = frame.copy()
    h_img, w_img = vis.shape[:2]
    ox, oy = w_img // 2, h_img // 2
    # Eixos do referencial da câmera.
    cv2.arrowedLine(vis, (ox, oy), (ox + 80, oy), (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(vis, (ox, oy), (ox, oy - 80), (0, 200, 0), 2, tipLength=0.2)
    cv2.putText(vis, "x", (ox + 85, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, "y", (ox - 5, oy - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if not resultado:
        return vis
    for c in resultado["caixas"]:
        cor = (0, 200, 0) if c["fonte"] == "gemini+opencv" else (0, 165, 255)
        pts = np.array(c["cantos_imagem_px"], dtype=np.int32)
        cv2.polylines(vis, [pts], True, cor, 2)
        cx, cy = map(int, c["centro_imagem_px"])
        cv2.line(vis, (ox, oy), (cx, cy), (180, 180, 180), 1)
        # Círculo magenta = centro dito pelo Gemini; cruz vermelha = centro final.
        g = c["gemini"]["centro"]
        cv2.circle(vis, (int(g["x"] + ox), int(oy - g["y"])), 7, (255, 0, 255), 2)
        cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        if c["angulo_graus"] is not None:  # seta na direção do lado maior
            a = np.radians(c["angulo_graus"])
            fim = (int(cx + 70 * np.cos(a)), int(cy - 70 * np.sin(a)))
            cv2.arrowedLine(vis, (cx, cy), fim, (255, 0, 0), 2, tipLength=0.25)
        rotulo = f'{c["id"]}: {c["nome"] or "?"} ({c["centro"]["x"]:.0f}, {c["centro"]["y"]:.0f})'
        if c["angulo_graus"] is not None:
            rotulo += f' {c["angulo_graus"]:.0f}deg'
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(vis, rotulo, (int(x0), max(20, int(y0) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(vis, rotulo, (int(x0), max(20, int(y0) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return vis


def salvar(resultado, frame):
    PASTA_SAIDA.mkdir(exist_ok=True)
    (PASTA_SAIDA / "ultimo_resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(PASTA_SAIDA / "ultimo_resultado.jpg"), desenhar(frame, resultado))


# --------------------------------------------------------------------------- main

class DetectorMovimento:
    """Diz quando a cena ficou parada e mudou desde a última análise."""

    def __init__(self, frames_estaveis=15, limiar=4.0):
        self.anterior = None
        self.analisado = None
        self.contador = 0
        self.frames_estaveis = frames_estaveis
        self.limiar = limiar

    @staticmethod
    def _reduzir(frame):
        return cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def atualizar(self, frame):
        p = self._reduzir(frame)
        if self.anterior is not None and cv2.absdiff(p, self.anterior).mean() < self.limiar / 4:
            self.contador += 1
        else:
            self.contador = 0
        self.anterior = p
        parada = self.contador >= self.frames_estaveis
        mudou = self.analisado is None or cv2.absdiff(p, self.analisado).mean() > self.limiar
        return parada and mudou

    def marcar_analisado(self, frame):
        self.analisado = self._reduzir(frame)


def rodar_imagem(caminho, args):
    frame = cv2.imread(caminho)
    if frame is None:
        raise SystemExit(f"Não consegui abrir {caminho}")
    gemini = AnalisadorGemini(args.modelo)
    resultado = analisar(frame, gemini, Segmentador(), carregar_homografia(), args.conf)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    salvar(resultado, frame)
    print(f"\nSalvo em {PASTA_SAIDA}")


def rodar_camera(args):
    gemini = AnalisadorGemini(args.modelo)
    segmentador = Segmentador()
    H = carregar_homografia()
    cam = Camera(args.camera, args.largura, args.altura)
    movimento = DetectorMovimento()

    estado = {"resultado": None, "frame_resultado": None, "ocupado": False, "erro": None}
    auto = True

    def tarefa(frame):
        try:
            r = analisar(frame, gemini, segmentador, H, args.conf)
            estado["resultado"], estado["frame_resultado"], estado["erro"] = r, frame, None
            print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)
            salvar(r, frame)
        except Exception as e:  # rede/API: mostra e continua rodando
            estado["erro"] = str(e)
            print(f"[erro] {e}", flush=True)
        finally:
            estado["ocupado"] = False

    def disparar(frame):
        estado["ocupado"] = True
        movimento.marcar_analisado(frame)
        threading.Thread(target=tarefa, args=(frame,), daemon=True).start()

    print("Aguardando câmera... (dica: com a mesa VAZIA aperte 'b' para capturar o fundo)")
    while (frame := cam.ler()) is None:
        time.sleep(0.05)

    try:
        while True:
            frame = cam.ler()
            pronto = movimento.atualizar(frame)
            if auto and pronto and not estado["ocupado"]:
                disparar(frame)

            vis = desenhar(frame, estado["resultado"])
            status = (f'{"ANALISANDO..." if estado["ocupado"] else "pronto"} | auto={"on" if auto else "off"}'
                      f' | fundo={"ok" if segmentador.fundo is not None else "nao"}'
                      f' | caixas={estado["resultado"]["total"] if estado["resultado"] else "-"}')
            if estado["erro"]:
                status += " | ERRO (ver terminal)"
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            escala = min(1.0, 1280 / vis.shape[1])
            cv2.imshow("Visao - remedios", cv2.resize(vis, None, fx=escala, fy=escala))
            if args.mascara:
                cv2.imshow("Mascara OpenCV", cv2.resize(segmentador.mascara(frame), None, fx=escala, fy=escala))

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" ") and not estado["ocupado"]:
                disparar(frame)
            elif tecla == ord("b"):
                segmentador.capturar_fundo(frame)
                print("Fundo capturado.")
            elif tecla == ord("a"):
                auto = not auto
            elif tecla == ord("s"):
                PASTA_SAIDA.mkdir(exist_ok=True)
                nome = PASTA_SAIDA / f"frame_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(nome), frame)
                print(f"Frame salvo: {nome}")
    finally:
        cam.fechar()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Localiza caixas de remédio (OpenCV + Gemini)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--imagem", help="analisa uma imagem em vez da câmera")
    ap.add_argument("--modelo", default=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    ap.add_argument("--largura", type=int, default=1920)
    ap.add_argument("--altura", type=int, default=1080)
    ap.add_argument("--conf", type=float, default=0.5, help="confiança mínima do Gemini")
    ap.add_argument("--mascara", action="store_true", help="mostra a máscara do OpenCV (debug)")
    args = ap.parse_args()

    if args.imagem:
        rodar_imagem(args.imagem, args)
    else:
        rodar_camera(args)


if __name__ == "__main__":
    main()
