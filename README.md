# only-vision

Sistema de visão para um braço robótico **SCARA**: localiza caixas de remédio sobre uma mesa branca e devolve, em JSON, o **centro de cada caixa e sua angulação em relação ao centro da câmera**, além do texto escrito nela.

Combina duas coisas:

- **Gemini (IA)** — encontra as caixas e devolve, para cada uma, o **centro**, o **eixo maior** (de onde sai o ângulo), a caixa delimitadora e o **texto** escrito nela.
- **OpenCV** — mede centro e ângulo de forma independente, com precisão de pixel, e serve de validação da IA (inclusive separando caixas encostadas).

![Exemplo de detecção](docs/exemplo.jpg)

## Como funciona

1. A câmera é lida continuamente em uma thread (o frame analisado é sempre o mais recente).
2. Quando a cena fica parada e mudou desde a última análise (ou ao apertar `ESPAÇO`), o frame é enviado ao Gemini.
3. O Gemini devolve, para cada caixa: `centro` (ponto), `eixo` (dois pontos nas pontas do eixo maior → ângulo), `box_2d`, `nome` e `texto`.
   O ângulo é pedido como dois pontos, e não como número, porque modelos de IA localizam pontos muito bem mas costumam errar o sentido de rotação quando o ângulo é pedido diretamente.
4. O OpenCV segmenta os objetos da mesa, separa caixas encostadas com *watershed* (usando o centro dado pelo Gemini como semente) e mede o retângulo rotacionado com `minAreaRect`.
5. **Fusão:** se o centro do OpenCV bate com o do Gemini (até 25% do lado menor da caixa), usa-se o valor do OpenCV, que tem precisão de pixel (`fonte: "gemini+opencv"`). Se não bate, ou se o OpenCV não achou a caixa, usa-se o do Gemini (`fonte: "gemini"` / `"gemini (opencv divergiu)"`). Os dois valores sempre ficam no JSON. Diferença de ângulo acima de 15° gera `alerta_angulo`.
6. O resultado é impresso no terminal e salvo em `saida/ultimo_resultado.json` e `saida/ultimo_resultado.jpg`.

Se o modelo principal do Gemini estiver sobrecarregado ou demorar mais que `GEMINI_TIMEOUT_S` (padrão 90 s), o código tenta automaticamente os modelos reserva (`MODELOS_RESERVA` em `vision_remedios.py`).

Na imagem anotada: eixos x/y no centro da câmera, cruz vermelha = centro final, círculo magenta = centro dado pelo Gemini, seta azul = direção do lado maior (ângulo).

## Instalação

Requer Python 3.10+.

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env` a partir do exemplo e coloque sua chave do [Google AI Studio](https://aistudio.google.com/apikey):

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TIMEOUT_S=90
```

## Uso

```bash
python vision_remedios.py                    # câmera 0
python vision_remedios.py --camera 1         # outra câmera
python vision_remedios.py --imagem foto.jpg  # analisa uma imagem e sai
python vision_remedios.py --mascara          # mostra a máscara do OpenCV (debug)
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--camera` | `0` | índice da câmera |
| `--imagem` | — | analisa um arquivo em vez da câmera |
| `--modelo` | `GEMINI_MODEL` do `.env` | modelo Gemini |
| `--largura` / `--altura` | `1920` / `1080` | resolução de captura |
| `--conf` | `0.5` | confiança mínima do Gemini para aceitar uma caixa |
| `--mascara` | — | abre uma janela com a máscara de segmentação |

### Teclas

| Tecla | Ação |
|---|---|
| `b` | captura o fundo — **faça isso com a mesa vazia** antes de colocar as caixas |
| `ESPAÇO` | analisa o frame atual |
| `a` | liga/desliga a análise automática |
| `s` | salva o frame atual em `saida/` |
| `q` / `ESC` | sai |

> Capturar o fundo (`b`) melhora muito a segmentação, principalmente com caixas brancas sobre a mesa branca.

## Saída

```json
{
  "timestamp": "2026-10-07T10:05:58.401",
  "modelo": "gemini-robotics-er-2-preview",
  "resolucao": [1920, 1080],
  "referencial": "origem no centro da câmera; x -> direita, y -> cima (px); angulo = lado maior da caixa vs eixo x, positivo = anti-horário",
  "tempo_gemini_s": 12.43,
  "calibrado_mm": false,
  "total": 1,
  "caixas": [
    {
      "id": 1,
      "nome": "AMOXICILINA",
      "texto": "AMOXICILINA | 500 mg - 20 comprimidos",
      "confianca": 1.0,
      "centro": {"x": -59.4, "y": -259.6},
      "angulo_graus": -12.0,
      "distancia_centro_px": 266.3,
      "direcao_graus": -102.9,
      "fonte": "gemini+opencv",
      "gemini": {
        "centro": {"x": -61.4, "y": -257.0},
        "angulo_graus": -12.6,
        "eixo": [{"x": -245.8, "y": -218.2}, {"x": 124.8, "y": -301.3}],
        "box_2d": [620, 362, 856, 574]
      },
      "opencv": {"centro": {"x": -59.4, "y": -259.6}, "angulo_graus": -12.0},
      "divergencia": {"centro_px": 3.3, "angulo_graus": 0.6, "alerta_angulo": false},
      "tamanho_px": [383.0, 181.0],
      "centro_imagem_px": [900.6, 799.6],
      "cantos_imagem_px": [[694.5, 848.3], [732.1, 671.3], [1106.7, 750.9], [1069.1, 927.9]]
    }
  ]
}
```

### Referencial

Todas as coordenadas principais são **relativas ao centro da câmera** (centro da imagem):

- **`centro`** `{x, y}` em pixels — x positivo para a **direita**, y positivo para **cima**.
- **`angulo_graus`** — ângulo do **lado maior** da caixa em relação ao eixo x, em [-90, 90); **positivo = anti-horário**. 0° = caixa deitada na horizontal; ±90° = em pé.
- **`distancia_centro_px`** / **`direcao_graus`** — a mesma posição em coordenadas polares (distância e direção do centro da câmera até a caixa; 0° = direita, 90° = cima).

### Demais campos

- **`fonte`** — de onde veio o valor final: `gemini+opencv` (os dois concordaram; usado o OpenCV), `gemini (opencv divergiu)` ou `gemini`.
- **`gemini`** / **`opencv`** — o valor que cada um mediu, para comparação.
- **`divergencia`** — diferença entre os dois (centro em px, ângulo em graus) e `alerta_angulo` se o ângulo divergir mais de 15°.
- **`centro_imagem_px`** / **`cantos_imagem_px`** — posição na imagem (origem no canto superior esquerdo, y para baixo), útil para desenhar.
- **`centro_mm`** — aparece quando existe calibração (ver abaixo).

## Calibração para milímetros

Para converter pixels em coordenadas do robô, crie um `calibracao.json` na raiz com a homografia `H` (pixel → mm):

```json
{ "H": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]] }
```

Com o arquivo presente, cada caixa passa a ter `centro_mm`. A ideia é gerar a `H` com um tabuleiro xadrez sobre a mesa (`cv2.findChessboardCorners` + `cv2.findHomography`).

## Montagem recomendada

- **Câmera** a ~80–90 cm, apontada para baixo, com foco e exposição **manuais/fixos**.
- **Iluminação** branca neutra/fria (**5000–6500 K**, **IRC ≥ 90**), **difusa** (softbox ou difusor leitoso). Duas fontes a ~45° em lados opostos ou um ring light ao redor da câmera eliminam praticamente todas as sombras. Evite luz colorida — distorce as cores das embalagens e prejudica a leitura do texto.
- Para caixas com acabamento brilhante, um **filtro polarizador** na lente reduz reflexos.

## Estrutura

```
vision_remedios.py   # código principal
requirements.txt
.env.example         # modelo do .env (a chave real não vai para o git)
docs/exemplo.jpg     # imagem de exemplo do README
saida/               # resultados gerados (ignorado pelo git)
```
