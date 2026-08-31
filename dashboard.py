"""Painel de desempenho do solver.

Grava o histórico de todas as partidas em JSON e redesenha um PNG a cada chute,
com a progressão da partida atual e a comparação com as anteriores.
"""

import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # sem GUI: só grava o arquivo

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HISTORICO_FILE = "historico.json"
GRAFICO_FILE = "desempenho.png"

# Limites e cores extraídos do bundle do próprio contexto.me:
#   distance < 300 -> verde | distance < 1500 -> amarelo | senão vermelho
# A UI mostra `distance + 1`, e é esse número que o solver guarda como rank,
# então os limites viram <= 300 e <= 1500.
LIMITE_VERDE = 300
LIMITE_AMARELO = 1500
VERDE, AMARELO, VERMELHO = "#00ba7c", "#ef7d31", "#f91880"
CINZA = "#3c4043"
VERDE_ESCURO = "#00875a"

MARCOS = [
    ("1º amarelo", LIMITE_AMARELO),
    ("1º verde", LIMITE_VERDE),
    ("top 10", 10),
    ("acertou", 1),
]


def cor_do_rank(rank):
    if rank <= LIMITE_VERDE:
        return VERDE
    if rank <= LIMITE_AMARELO:
        return AMARELO
    return VERMELHO


# ---------------------------------------------------------------- histórico


def carregar_historico():
    if os.path.exists(HISTORICO_FILE):
        with open(HISTORICO_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"jogos": []}


def salvar_historico(historico):
    with open(HISTORICO_FILE, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=1)


def iniciar_jogo(historico, jogo_id):
    """Cria o registro da partida. Re-executar no mesmo dia sobrescreve o anterior,
    já que é a mesma palavra secreta e o solver recomeça do zero."""
    historico["jogos"] = [j for j in historico["jogos"] if j["id"] != jogo_id]
    jogo = {
        "id": jogo_id,
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "chutes": [],
        "invalidos": 0,
        "repetidos": 0,
        "resolvido": False,
        "palavra": None,
    }
    historico["jogos"].append(jogo)
    return jogo


def registrar_chute(jogo, palavra, rank, estrategia):
    jogo["chutes"].append(
        {
            "n": len(jogo["chutes"]) + 1,
            "palavra": palavra,
            "rank": rank,
            "estrategia": estrategia,
        }
    )
    if rank == 1:
        jogo["resolvido"] = True
        jogo["palavra"] = palavra


# ----------------------------------------------------------------- métricas


def _ranks(jogo):
    return [c["rank"] for c in jogo["chutes"]]


def _melhor_acumulado(jogo):
    """Melhor rank visto até cada chute (escada monotônica decrescente)."""
    melhor, saida = float("inf"), []
    for r in _ranks(jogo):
        melhor = min(melhor, r)
        saida.append(melhor)
    return saida


def marcos_do_jogo(jogo):
    """{nome do marco: número do chute em que foi atingido pela primeira vez}."""
    atingidos = {}
    for nome, limite in MARCOS:
        for c in jogo["chutes"]:
            if c["rank"] <= limite:
                atingidos[nome] = c["n"]
                break
    return atingidos


def _desperdicados(jogo):
    """Chutes que não melhoraram o melhor rank — o custo da estratégia."""
    melhor = _melhor_acumulado(jogo)
    return sum(1 for i in range(1, len(melhor)) if melhor[i] == melhor[i - 1])


def resumo(historico, jogo_atual):
    anteriores = [j for j in historico["jogos"] if j["id"] != jogo_atual["id"]]
    resolvidos = [len(j["chutes"]) for j in anteriores if j["resolvido"]]
    total_chutes = sum(len(j["chutes"]) for j in anteriores) or 1
    return {
        "partidas": len(anteriores),
        "resolvidas": len(resolvidos),
        "taxa": len(resolvidos) / len(anteriores) if anteriores else 0.0,
        "mediana": float(np.median(resolvidos)) if resolvidos else None,
        "p90": float(np.percentile(resolvidos, 90)) if resolvidos else None,
        "desperdicio": sum(_desperdicados(j) for j in anteriores) / total_chutes,
    }


# ------------------------------------------------------------------ painéis


def _painel_progressao(ax, jogo):
    ranks = _ranks(jogo)
    ax.set_title(
        f"Progressão da partida — {jogo['id']}", fontsize=13, weight="bold", loc="left"
    )
    ax.set_xlabel("nº do chute")
    ax.set_ylabel("posição (log, menor = melhor)")

    if not ranks:
        ax.text(0.5, 0.5, "aguardando o primeiro chute…", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    xs = list(range(1, len(ranks) + 1))
    topo = max(max(ranks), LIMITE_AMARELO + 1) * 1.25

    # faixas de cor do próprio jogo
    ax.axhspan(1, LIMITE_VERDE, color=VERDE, alpha=0.13, zorder=0)
    ax.axhspan(LIMITE_VERDE, LIMITE_AMARELO, color=AMARELO, alpha=0.13, zorder=0)
    ax.axhspan(LIMITE_AMARELO, topo, color=VERMELHO, alpha=0.10, zorder=0)

    melhor = _melhor_acumulado(jogo)
    ax.step(xs, melhor, where="post", color=CINZA, lw=2, zorder=3, label="melhor até aqui")
    ax.scatter(
        xs,
        ranks,
        c=[cor_do_rank(r) for r in ranks],
        s=55,
        edgecolor="white",
        lw=0.8,
        zorder=4,
    )

    # marca a entrada no verde, que é o ponto de virada da partida
    entrada = marcos_do_jogo(jogo).get("1º verde")
    if entrada:
        ax.axvline(entrada, color=VERDE, ls="--", lw=1.5, zorder=2)
        ax.annotate(
            f"entrou no verde\nno chute {entrada}",
            xy=(entrada, LIMITE_VERDE),
            xytext=(6, 14),
            textcoords="offset points",
            fontsize=9,
            color=VERDE,
            weight="bold",
        )

    # rotula so o primeiro chute, a resposta e os recordes que representam um
    # salto grande — rotular todo recorde faz os nomes colidirem
    for i, r in enumerate(ranks):
        recorde = i == 0 or melhor[i] < melhor[i - 1]
        salto = i == 0 or r == 1 or melhor[i] <= 0.6 * melhor[i - 1]
        if recorde and salto:
            ax.annotate(
                jogo["chutes"][i]["palavra"],
                xy=(xs[i], r),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=CINZA,
            )

    ax.set_yscale("log")
    ax.set_ylim(topo, 0.8)  # invertido: rank 1 no topo
    ax.set_xlim(0.4, len(ranks) + 0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25, zorder=1)
    # a trajetória sobe da esquerda para a direita, então o canto inferior
    # direito é o único que fica livre de forma confiável
    ax.legend(
        handles=[
            plt.Line2D([], [], color=CINZA, lw=2, label="melhor até aqui"),
            Patch(color=VERDE, alpha=0.4, label=f"verde (≤ {LIMITE_VERDE})"),
            Patch(color=AMARELO, alpha=0.4, label=f"amarelo (≤ {LIMITE_AMARELO})"),
            Patch(color=VERMELHO, alpha=0.4, label="vermelho"),
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.92,
    )


def _painel_marcos(ax, historico, jogo_atual):
    """Quantos chutes até cada estágio, comparado com a mediana das partidas anteriores."""
    ax.set_title("Chutes até cada marco", fontsize=12, weight="bold", loc="left")

    atuais = marcos_do_jogo(jogo_atual)
    anteriores = [j for j in historico["jogos"] if j["id"] != jogo_atual["id"]]

    nomes = [nome for nome, _ in MARCOS]
    ys = np.arange(len(nomes))
    valores = [atuais.get(n, 0) for n in nomes]
    cores = [AMARELO, VERDE, VERDE, VERDE_ESCURO]

    ax.barh(ys, valores, color=cores, height=0.55, zorder=3)

    medianas = []
    for nome in nomes:
        vals = [marcos_do_jogo(j).get(nome) for j in anteriores]
        vals = [v for v in vals if v]
        medianas.append(float(np.median(vals)) if vals else None)

    limite = max([v for v in valores if v] + [m for m in medianas if m] + [1])

    for y, (v, med) in enumerate(zip(valores, medianas)):
        if not v:
            ax.text(0.3, y, " não atingido", va="center", fontsize=9, color="#999")
        elif v >= 0.15 * limite:
            # dentro da barra: fora colidiria com o marcador da mediana, que
            # costuma cair bem perto do valor atual
            ax.text(
                v,
                y,
                f"{v} ",
                va="center",
                ha="right",
                fontsize=10,
                weight="bold",
                color="white",
                zorder=6,
            )
        else:
            ax.text(v, y, f" {v}", va="center", fontsize=10, weight="bold", zorder=6)
        if med:
            ax.scatter([med], [y], marker="|", s=350, color=CINZA, lw=2, zorder=5)

    if any(m for m in medianas):
        ax.scatter(
            [], [], marker="|", s=200, color=CINZA, lw=2, label="mediana anterior"
        )
        # as barras crescem para a direita e a maior fica embaixo, entao o
        # canto superior direito e o que sobra
        ax.legend(fontsize=8.5, loc="upper right")

    # folga a direita para o numero no fim da barra nao ser cortado
    ax.set_xlim(0, limite * 1.18)

    ax.set_yticks(ys)
    ax.set_yticklabels(nomes, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("nº do chute")
    ax.grid(axis="x", alpha=0.25, zorder=0)


def _painel_historico(ax, historico, jogo_atual):
    ax.set_title("Histórico de partidas", fontsize=12, weight="bold", loc="left")

    jogos = historico["jogos"]
    if not jogos:
        ax.axis("off")
        return

    xs = np.arange(len(jogos))
    alturas = [len(j["chutes"]) for j in jogos]
    cores = [
        VERDE if j["resolvido"] else "#cccccc" for j in jogos
    ]
    bordas = ["none" if j["resolvido"] else VERMELHO for j in jogos]
    hachuras = ["" if j["resolvido"] else "//" for j in jogos]
    ax.bar(
        xs,
        alturas,
        color=cores,
        edgecolor=bordas,
        hatch=hachuras,
        lw=1.2,
        zorder=3,
    )

    r = resumo(historico, jogo_atual)
    if r["mediana"]:
        ax.axhline(
            r["mediana"],
            color=CINZA,
            ls="--",
            lw=1.4,
            zorder=4,
            label=f"mediana {r['mediana']:.0f}",
        )
        ax.legend(fontsize=8.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([j["id"][5:] for j in jogos], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("chutes na partida")
    ax.grid(axis="y", alpha=0.25, zorder=0)


def _rodape(fig, historico, jogo_atual):
    r = resumo(historico, jogo_atual)
    partes = [
        f"partidas anteriores: {r['partidas']}",
        f"resolvidas: {r['resolvidas']} ({r['taxa']:.0%})",
        f"mediana: {r['mediana']:.0f}" if r["mediana"] else "mediana: --",
        f"p90: {r['p90']:.0f}" if r["p90"] else "p90: --",
        f"chutes sem ganho: {r['desperdicio']:.0%}",
    ]
    fig.text(
        0.01,
        -0.01,
        "   ·   ".join(partes),
        fontsize=9.5,
        family="monospace",
        color=CINZA,
        va="top",
    )


def atualizar_grafico(historico, jogo_atual, caminho=GRAFICO_FILE):
    """Redesenha o PNG do zero. Chamado a cada chute — é barato o bastante
    perto do custo de um chute no site."""
    fig = plt.figure(figsize=(13.5, 8.5), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.35, wspace=0.28)

    _painel_progressao(fig.add_subplot(gs[0, :]), jogo_atual)
    _painel_marcos(fig.add_subplot(gs[1, 0]), historico, jogo_atual)
    _painel_historico(fig.add_subplot(gs[1, 1]), historico, jogo_atual)
    _rodape(fig, historico, jogo_atual)

    n = len(jogo_atual["chutes"])
    melhor = min(_ranks(jogo_atual)) if n else None
    estado = (
        f"resolvido: {jogo_atual['palavra']}"
        if jogo_atual["resolvido"]
        else (f"melhor até agora: {melhor}" if melhor else "iniciando")
    )
    invalidas = jogo_atual["invalidos"]
    rejeitadas = f"{invalidas} palavra{'s' if invalidas != 1 else ''} rejeitada"
    rejeitadas += "s" if invalidas != 1 else ""
    fig.suptitle(
        f"Contexto solver — {n} chutes · {estado} · {rejeitadas}",
        fontsize=14,
        weight="bold",
        x=0.01,
        ha="left",
    )

    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
