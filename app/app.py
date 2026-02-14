import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "SBERT_nli.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
ckpt = torch.load(MODEL_PATH, map_location="cpu")

word2idx = ckpt["word2idx"]
config = ckpt["config"]

PAD = word2idx["[PAD]"]
CLS = word2idx["[CLS]"]
SEP = word2idx["[SEP]"]
UNK = word2idx["[UNK]"]

max_len = int(config["max_len"])
hidden = int(config["hidden"])
id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}


class MiniBERT_EncoderOnly(nn.Module):
    def __init__(self, vocab_size, hidden=256, max_len=128, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.pos_embed = nn.Embedding(max_len, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, input_ids, attn_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        x = self.drop(self.ln(x))
        src_key_padding_mask = (attn_mask == 0)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask) 
        return h

def mean_pool(token_embeds, attn_mask):
    mask = attn_mask.unsqueeze(-1).float()
    summed = (token_embeds * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

class SBERTSoftmax(nn.Module):
    def __init__(self, encoder, hidden):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden * 3, 3)

    def forward(self, ids_a, mask_a, ids_b, mask_b):
        h_a = self.encoder(ids_a, mask_a)
        h_b = self.encoder(ids_b, mask_b)
        u = mean_pool(h_a, mask_a)
        v = mean_pool(h_b, mask_b)
        feats = torch.cat([u, v, torch.abs(u - v)], dim=-1)
        logits = self.classifier(feats)
        return logits, u, v

encoder = MiniBERT_EncoderOnly(
    vocab_size=len(word2idx),
    hidden=hidden,
    max_len=max_len,
    n_layers=int(config.get("n_layers", 4)),
    n_heads=int(config.get("n_heads", 4))).to(DEVICE)

encoder.load_state_dict(ckpt["encoder_state"], strict=True)

model = SBERTSoftmax(encoder, hidden=hidden).to(DEVICE)
model.classifier.load_state_dict(ckpt["classifier_state"], strict=True)
model.eval()


def encode_sentence(text: str):
    text = (text or "").strip()
    toks = [word2idx.get(w, UNK) for w in text.lower().split()][: max_len - 2]
    seq = [CLS] + toks + [SEP]
    attn = [1] * len(seq)

    pad_len = max_len - len(seq)
    seq += [PAD] * pad_len
    attn += [0] * pad_len

    ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)
    mask = torch.tensor([attn], dtype=torch.long, device=DEVICE)
    return ids, mask

def predict(sent_a: str, sent_b: str):
    ids_a, mask_a = encode_sentence(sent_a)
    ids_b, mask_b = encode_sentence(sent_b)

    with torch.no_grad():
        logits, u, v = model(ids_a, mask_a, ids_b, mask_b)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probs).item())
        cos = float(F.cosine_similarity(u, v).item())

    return {
        "label": id2label[pred_id],
        "cos": cos,
        "probs": {
            "entailment": float(probs[0].item()),
            "neutral": float(probs[1].item()),
            "contradiction": float(probs[2].item()),
        },
    }


THEME = dbc.themes.CYBORG
INTER_FONT = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap"

COLOR = {
    "entailment": "#2ecc71",
    "neutral": "#f1c40f",
    "contradiction": "#e74c3c",
    "muted": "rgba(255,255,255,0.70)",
    "muted2": "rgba(255,255,255,0.55)"}

def label_badge(label: str):
    label = (label or "").lower()
    c = COLOR.get(label, "#9b59b6")
    return dbc.Badge(
        label.upper(),
        color="dark",
        style={
            "background": c,
            "padding": "8px 12px",
            "borderRadius": "999px",
            "fontWeight": "700",
            "letterSpacing": "0.6px",
        },
    )

def cosine_to_0_1(cos: float) -> float:
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))

def fmt_prob(p: float) -> str:
    return f"{p:.4f}"


app = Dash(__name__, external_stylesheets=[THEME, INTER_FONT])
server = app.server

APP_BG = {
    "minHeight": "100vh",
    "paddingBottom": "32px",
    "background": "radial-gradient(1200px 600px at 10% 0%, rgba(88,101,242,0.25), transparent 55%),"
                  "radial-gradient(1000px 500px at 90% 10%, rgba(46,204,113,0.18), transparent 55%),"
                  "linear-gradient(180deg, #0b1020 0%, #0a0f1a 60%, #070b12 100%)",
    "fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial",
}

CARD = {
    "borderRadius": "18px",
    "background": "rgba(255,255,255,0.06)",
    "border": "1px solid rgba(255,255,255,0.10)",
    "boxShadow": "0 16px 50px rgba(0,0,0,0.45)",
    "backdropFilter": "blur(10px)",
    "color": "rgba(255,255,255,0.92)"
}

TEXTAREA = {
    "width": "100%",
    "height": "150px",
    "borderRadius": "14px",
    "padding": "12px",
    "resize": "vertical",
    "background": "rgba(10,15,26,0.70)",
    "color": "rgba(255,255,255,0.92)",
    "border": "1px solid rgba(255,255,255,0.14)",
    "outline": "none",
}

SMALL_META = {
    "color": COLOR["muted2"],
    "fontSize": "13px",
}

app.layout = html.Div(
    style=APP_BG,
    children=[
        dbc.Container(
            fluid=True,
            style={"maxWidth": "1140px", "paddingTop": "26px"},
            children=[
                # Header
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "gap": "14px",
                        "flexWrap": "wrap",
                        "marginBottom": "18px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    "Text Similarity Web App",
                                    style={"color": COLOR["muted2"], "fontWeight": "600", "letterSpacing": "0.3px"},
                                ),
                                html.H1(
                                    "SBERT NLI Demo",
                                    style={"margin": "6px 0 0 0", "fontWeight": "700"},
                                ),
                                html.Div(
                                    "Compare sentence meaning via cosine similarity + NLI (entailment / neutral / contradiction).",
                                    style={"color": COLOR["muted"], "marginTop": "6px"},
                                ),
                            ]
                        ),
                        dbc.Card(
                            style={**CARD, "padding": "14px 16px", "minWidth": "280px"},
                            children=[
                                html.Div("Model Info", style={"fontWeight": "700", "marginBottom": "8px"}),
                                html.Div(
                                    [
                                        html.Span("vocab=", style=SMALL_META),
                                        html.Span(str(len(word2idx)), style={"fontWeight": "700"}),
                                        html.Span("  •  ", style=SMALL_META),
                                        html.Span("max_len=", style=SMALL_META),
                                        html.Span(str(max_len), style={"fontWeight": "700"}),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Span("hidden=", style=SMALL_META),
                                        html.Span(str(hidden), style={"fontWeight": "700"}),
                                        html.Span("  •  ", style=SMALL_META),
                                        html.Span("device=", style=SMALL_META),
                                        html.Span(str(DEVICE), style={"fontWeight": "700"}),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),

                # Inputs
                dbc.Row(
                    className="g-3",
                    children=[
                        dbc.Col(
                            md=6,
                            children=dbc.Card(
                                style=CARD,
                                body=True,
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
                                        children=[
                                            html.H5("Sentence A", style={"margin": 0, "fontWeight": "700"}),
                                            html.Div("Premise / Text A", style=SMALL_META),
                                        ],
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    dcc.Textarea(
                                        id="sent-a",
                                        value="A man is playing guitar.",
                                        style=TEXTAREA,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Div("Try clear sentences.", style=SMALL_META),
                                ],
                            ),
                        ),
                        dbc.Col(
                            md=6,
                            children=dbc.Card(
                                style=CARD,
                                body=True,
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
                                        children=[
                                            html.H5("Sentence B", style={"margin": 0, "fontWeight": "700"}),
                                            html.Div("Hypothesis / Text B", style=SMALL_META),
                                        ],
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    dcc.Textarea(
                                        id="sent-b",
                                        value="A person is performing music.",
                                        style=TEXTAREA,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Div("Try paraphrases to see cosine similarity change.", style=SMALL_META),
                                ],
                            ),
                        ),
                    ],
                ),

                html.Div(style={"height": "14px"}),

                # Action row
                dbc.Row(
                    className="g-3",
                    align="center",
                    children=[
                        dbc.Col(
                            md=8,
                            children=dbc.Card(
                                style={**CARD, "padding": "14px 16px"},
                                children=[
                                    html.Div(
                                        style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "10px"},
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div("Compute inference", style={"fontWeight": "700"}),
                                                    html.Div("Outputs: cosine similarity + NLI label + probabilities", style=SMALL_META),
                                                ]
                                            ),
                                            dbc.Button(
                                                "Run Prediction",
                                                id="btn",
                                                n_clicks=0,
                                                color="primary",
                                                size="lg",
                                                style={
                                                    "borderRadius": "14px",
                                                    "padding": "10px 16px",
                                                    "fontWeight": "700",
                                                },
                                            ),
                                        ],
                                    ),
                                    html.Div(id="err", style={"color": "#ff6b6b", "marginTop": "10px"}),
                                ],
                            ),
                        ),
                        dbc.Col(
                            md=4,
                            children=dbc.Card(
                                style={**CARD, "padding": "14px 16px"},
                                children=[
                                    html.Div("Quick Examples", style={"fontWeight": "700", "marginBottom": "8px", "color": "rgba(255,255,255,0.95)"}),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("Similar", id="ex-sim", outline=True, color="light", size="sm"),
                                            dbc.Button("Neutral", id="ex-neu", outline=True, color="light", size="sm"),
                                            dbc.Button("Contradict", id="ex-con", outline=True, color="light", size="sm"),
                                        ],
                                        size="sm",
                                    ),
                                    html.Div(style={"height": "8px"}),
                                    html.Div("Click to auto-fill inputs.", style=SMALL_META),
                                ],
                            ),
                        ),
                    ],
                ),

                html.Div(style={"height": "14px"}),

                # Results
                dbc.Row(
                    className="g-3",
                    children=[
                        dbc.Col(
                            md=5,
                            children=dbc.Card(
                                style=CARD,
                                body=True,
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                                        children=[
                                            html.H4("Result", style={"margin": 0, "fontWeight": "800"}),
                                            html.Div(id="pred-badge"),
                                        ],
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Div(id="pred-label", style={"fontSize": "18px", "color": COLOR["muted"]}),
                                    html.Div(style={"height": "12px"}),
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline"},
                                        children=[
                                            html.Div("Cosine Similarity", style={"fontWeight": "700"}),
                                            html.Div(id="cos-text", style={"fontFamily": "ui-monospace, Menlo, Consolas, monospace"}),
                                        ],
                                    ),
                                    html.Div(style={"height": "6px"}),
                                    dbc.Progress(id="cos-bar", value=50, striped=True, animated=True, style={"height": "12px", "borderRadius": "999px"}),
                                    html.Div(style={"height": "10px"}),
                                    html.Div(id="cos-hint", style=SMALL_META),
                                ],
                            ),
                        ),
                        dbc.Col(
                            md=7,
                            children=dbc.Card(
                                style=CARD,
                                body=True,
                                children=[
                                    html.H4("Class Probabilities", style={"fontWeight": "800"}),
                                    html.Div(style={"height": "10px"}),

                                    html.Div("Entailment", style={"fontWeight": "700"}),
                                    dbc.Progress(id="p-ent", value=0, color="success", style={"height": "12px", "borderRadius": "999px"}),
                                    html.Div(id="t-ent", style=SMALL_META),
                                    html.Div(style={"height": "10px"}),

                                    html.Div("Neutral", style={"fontWeight": "700"}),
                                    dbc.Progress(id="p-neu", value=0, color="warning", style={"height": "12px", "borderRadius": "999px"}),
                                    html.Div(id="t-neu", style=SMALL_META),
                                    html.Div(style={"height": "10px"}),

                                    html.Div("Contradiction", style={"fontWeight": "700"}),
                                    dbc.Progress(id="p-con", value=0, color="danger", style={"height": "12px", "borderRadius": "999px"}),
                                    html.Div(id="t-con", style=SMALL_META),

                                    html.Div(style={"height": "14px"}),
                                    html.Div(
                                        id="raw-json",
                                        style={
                                            "display": "none",
                                        },
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),

                html.Div(style={"height": "18px"}),

                # Footer
                dbc.Card(
                    style={**CARD, "padding": "14px 16px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "8px"},
                            children=[
                                html.Div("Built with Dash • SBERT (from scratch) • SNLI", style=SMALL_META),
                                html.Div("Outputs: cosine(u,v) + softmax(Wᵀ[u,v,|u−v|])", style=SMALL_META),
                            ],
                        )
                    ],
                ),
            ],
        )
    ],
)


@app.callback(
    Output("sent-a", "value"),
    Output("sent-b", "value"),
    Input("ex-sim", "n_clicks"),
    Input("ex-neu", "n_clicks"),
    Input("ex-con", "n_clicks"),
    prevent_initial_call=True,
)
def fill_examples(n1, n2, n3):
    ctx = __import__("dash").callback_context
    if not ctx.triggered:
        raise __import__("dash").exceptions.PreventUpdate

    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "ex-sim":
        return "A man is playing guitar.", "A person is performing music."
    if trig == "ex-neu":
        return "A dog is running in the park.", "A person is reading a book indoors."
    return "A woman is cooking dinner.", "Nobody is preparing any food."

@app.callback(
    Output("pred-badge", "children"),
    Output("pred-label", "children"),
    Output("cos-text", "children"),
    Output("cos-bar", "value"),
    Output("cos-bar", "color"),
    Output("cos-hint", "children"),
    Output("p-ent", "value"),
    Output("p-neu", "value"),
    Output("p-con", "value"),
    Output("t-ent", "children"),
    Output("t-neu", "children"),
    Output("t-con", "children"),
    Output("err", "children"),
    Input("btn", "n_clicks"),
    State("sent-a", "value"),
    State("sent-b", "value"),
)
def run_predict(n, a, b):
    if n is None:
        raise __import__("dash").exceptions.PreventUpdate

    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return "", "", "", 50, "info", "", 0, 0, 0, "", "", "", "Please enter both sentences."

    try:
        out = predict(a, b)
        label = out["label"]
        cos = float(out["cos"])
        probs = out["probs"]

    
        cos01 = cosine_to_0_1(cos)          # [0,1]
        cos_pct = cos01 * 100.0             # [0,100]
        if cos > 0.6:
            cos_color = "success"
            cos_hint = "High similarity: sentences are strongly related."
        elif cos > 0.2:
            cos_color = "info"
            cos_hint = "Moderate similarity: sentences are related but not identical."
        elif cos > -0.2:
            cos_color = "warning"
            cos_hint = "Low similarity: sentences weakly related."
        else:
            cos_color = "danger"
            cos_hint = "Negative similarity: sentences likely unrelated/opposite."

        # probability bars
        ent = probs["entailment"] * 100
        neu = probs["neutral"] * 100
        con = probs["contradiction"] * 100

        return (
            label_badge(label),
            f"Predicted NLI label: {label}",
            f"{cos:.4f}",
            cos_pct,
            cos_color,
            cos_hint,
            ent,
            neu,
            con,
            f"p = {fmt_prob(probs['entailment'])}",
            f"p = {fmt_prob(probs['neutral'])}",
            f"p = {fmt_prob(probs['contradiction'])}",
            "",
        )
    except Exception as e:
        return "", "", "", 50, "info", "", 0, 0, 0, "", "", "", f"Error: {repr(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)

