print("Importando 0/17")
import os
print("Importando 1/17")
import torch
print("Importando 2/17")
import random
print("Importando 3/17")
import numpy as np
print("Importando 4/17")
import dashboard
print("Importando 5/17")
from datetime import date
print("Importando 6/17")
from selenium import webdriver
print("Importando 7/17")
from nltk.stem import PorterStemmer
print("Importando 8/17")
from IPython.display import clear_output
print("Importando 9/17")
from selenium.webdriver.common.by import By
print("Importando 10/17")
from selenium.webdriver.common.keys import Keys
print("Importando 11/17")
from transformers import BertTokenizer, BertModel
print("Importando 12/17")
from selenium.webdriver.chrome.service import Service
print("Importando 13/17")
from selenium.webdriver.support.ui import WebDriverWait
print("Importando 14/17")
from webdriver_manager.chrome import ChromeDriverManager
print("Importando 15/17")
from selenium.common.exceptions import NoSuchElementException, TimeoutException
print("Importando 16/17")
from selenium.webdriver.support import expected_conditions as EC
print("Importando 17/17")

print("Iniciando")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://contexto.me/en/")

print("Driver")

wait = WebDriverWait(driver, 20)

print("Driver 2.0")

def abrir_partida_diaria():
    """Entra na partida do dia.

    O /en/daily so existe via SPA (acessar a URL direto redireciona pra home) e o
    iframe de anuncio do Google costuma cobrir o link, o que faz o clique nativo
    estourar ElementClickInterceptedException. O clique via JS ignora o overlay.
    """
    link = wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "a[href='/en/daily']"))
    )
    driver.execute_script("arguments[0].click();", link)

    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.word")))
        return
    except TimeoutException:
        pass

    # ultimo recurso: clique nativo, caso o site passe a exigir evento real
    link = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='/en/daily']"))
    )
    link.click()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.word")))


abrir_partida_diaria()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
print("BERT Carregado com sucesso!")


def encode_words(words):
    """Mean pooling over real tokens only, so a word's vector is the same
    alone or inside a batch ([PAD]/[CLS]/[SEP] are excluded)."""
    inputs = tokenizer(
        words,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_special_tokens_mask=True,
    )
    special_mask = inputs.pop("special_tokens_mask")

    with torch.no_grad():
        outputs = model(**inputs)

    pool_mask = (inputs["attention_mask"] * (1 - special_mask)).unsqueeze(-1).float()
    soma = (outputs.last_hidden_state * pool_mask).sum(dim=1)
    qtd = pool_mask.sum(dim=1).clamp(min=1e-9)
    return (soma / qtd).numpy()


embeddings = {}
memory = {}

# _v2: the old file was pooled over [PAD]/[CLS]/[SEP] and is invalid
EMBEDDINGS_FILE = "embeddingsValues.npz"

if os.path.exists(EMBEDDINGS_FILE):
    print("Arquivo de embeddings encontrado.")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    candidatos_raw = data["candidatos"]
    matriz_raw = data["matriz"]

    # dedup keeping the first occurrence
    seen_load = {}
    for i, w in enumerate(candidatos_raw):
        w_str = str(w)
        if w_str not in seen_load:
            seen_load[w_str] = i
    indices_unicos = sorted(seen_load.values())
    candidatos_lista = [str(candidatos_raw[i]) for i in indices_unicos]
    matriz_candidatos = matriz_raw[indices_unicos]

    for word, vec in zip(candidatos_lista, matriz_candidatos):
        embeddings[word] = vec
    print(
        f"Pronto! {len(candidatos_lista)} candidatos carregados ({len(candidatos_raw) - len(candidatos_lista)} duplicatas removidas)."
    )
else:
    print("Arquivo nao encontrado. Pre-computando embeddings")
    todos_candidatos = [w for w in tokenizer.vocab.keys() if w.isalpha() and len(w) > 4]

    batch_size = 64
    candidatos_lista = []
    matriz_candidatos = []

    for i in range(0, len(todos_candidatos), batch_size):
        batch = todos_candidatos[i : i + batch_size]
        embs = encode_words(batch)
        for j, word in enumerate(batch):
            candidatos_lista.append(word)
            matriz_candidatos.append(embs[j])
            embeddings[word] = embs[j]

    matriz_candidatos = np.array(matriz_candidatos)
    np.savez(EMBEDDINGS_FILE, candidatos=candidatos_lista, matriz=matriz_candidatos)
    print(
        f"Pronto! {len(candidatos_lista)} candidatos pre-computados e salvos em '{EMBEDDINGS_FILE}'."
    )

stemmer = PorterStemmer()
candidatos_stems = [stemmer.stem(w) for w in candidatos_lista]
print(f"Radicais pre-computados para {len(candidatos_stems)} candidatos.")

# normalized once, so cosine is just a dot product
matriz_norm = matriz_candidatos / np.linalg.norm(
    matriz_candidatos, axis=1, keepdims=True
)

temas_exploratorios = [
    "object",
    "place",
    "person",
    "animal",
    "emotion",
    "food",
    "vehicle",
    "technology",
    "family",
    "music",
    "clothing",
    "plant",
    "profession",
    "color",
]

palavras_usadas = set()
stems_usados = set()
n_sugestoes = 10

# palavras que o site recusa (desconhecidas ou comuns demais). Persistidas entre
# execucoes para nunca mais gastar um chute com elas.
PALAVRAS_INVALIDAS_FILE = "palavras_invalidas.txt"
palavras_invalidas = set()

if os.path.exists(PALAVRAS_INVALIDAS_FILE):
    with open(PALAVRAS_INVALIDAS_FILE, encoding="utf-8") as f:
        palavras_invalidas = {linha.strip().lower() for linha in f if linha.strip()}
    print(f"{len(palavras_invalidas)} palavras invalidas carregadas de '{PALAVRAS_INVALIDAS_FILE}'.")


def registrar_palavra_invalida(palavra, motivo):
    """Marca a palavra como inutilizavel e grava no arquivo imediatamente."""
    palavra = str(palavra).strip().lower()
    palavras_usadas.add(palavra)
    jogo_atual["invalidos"] += 1
    if palavra in palavras_invalidas:
        return
    palavras_invalidas.add(palavra)
    with open(PALAVRAS_INVALIDAS_FILE, "a", encoding="utf-8") as f:
        print(palavra, file=f)
    print(f"[invalida ({motivo}): {palavra}]")


def _classificar_mensagem(texto):
    """Le o balao de aviso do site: 'repetida' | 'desconhecida' | 'comum' | None."""
    t = str(texto).strip().lower().replace("’", "'")
    if not t:
        return None
    if "guessed" in t:
        return "repetida"
    if "know this word" in t:
        return "desconhecida"
    if "too common" in t:
        return "comum"
    return None


temas_exploratorios = [t for t in temas_exploratorios if t not in palavras_invalidas]

historico = dashboard.carregar_historico()
jogo_atual = dashboard.iniciar_jogo(historico, date.today().isoformat())
# de qual ramo de sugerir_proximo() veio o chute corrente, para o log por chute
ultima_estrategia = "tema"

# below 4 guesses the rank correlation is noise, so stick to the centroid
MIN_CHUTES_CONSISTENCIA = 4
# each new rank is very informative here, and the suggestions are near-duplicates
N_SUGESTOES_CONSISTENCIA = 3


def _ranks_por_linha(S):
    """Ordinal ranks per row (0 = smallest)."""
    return S.argsort(axis=1).argsort(axis=1).astype(np.float64)


def _corr_linhas(R, y):
    """Pearson between each row of R and y."""
    Rc = R - R.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    num = Rc @ yc
    den = np.linalg.norm(Rc, axis=1) * np.linalg.norm(yc)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def _normalizar_01(v):
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def score_consistencia(matriz_cand_norm, vetores_chutes, ranks_chutes):
    """Spearman between sim(candidate, guesses) and -rank of the guesses.

    If the candidate were the target, its similarity to each guess would drop as
    the rank grows. Unlike the centroid, bad guesses act as repulsion. Uses only
    the order of the similarities, which is more reliable than BERT's raw cosine.
    """
    S = matriz_cand_norm @ vetores_chutes.T
    R = _ranks_por_linha(S)
    y = _ranks_por_linha(-ranks_chutes[None, :])[0]
    return _corr_linhas(R, y)


def _top_n(scores, n):
    """Indices of the n highest scores, best first."""
    n = min(n, len(scores))
    if n >= len(scores):
        idx = np.arange(len(scores))
    else:
        idx = np.argpartition(scores, -n)[-n:]
    return idx[np.argsort(-scores[idx])]


def _marcar(palavra):
    palavras_usadas.add(palavra)
    stems_usados.add(stemmer.stem(palavra))


def get_embedding(word):
    if word in embeddings:
        return embeddings[word]

    emb = encode_words([word])[0]
    embeddings[word] = emb
    return emb


def adicionar_chute(palavra, score):
    memory[palavra] = score
    _marcar(palavra)
    get_embedding(palavra)


def _sugerir_por_centroide(cands, M):
    """Early phase: neighbors of the weighted mean of the best guesses."""
    perfeitos = sorted(
        [(p, s) for p, s in memory.items() if s < 50], key=lambda x: x[1]
    )
    otimos = sorted(
        [(p, s) for p, s in memory.items() if 50 <= s < 200], key=lambda x: x[1]
    )

    MIN_BASE = 10
    base = []
    for group in [perfeitos, otimos]:
        if len(base) >= MIN_BASE:
            break
        base.extend(list(group)[: MIN_BASE - len(base)])

    if not base:
        base = sorted(memory.items(), key=lambda x: x[1])[:MIN_BASE]

    vetores = np.array([get_embedding(p) for p, _ in base])
    pesos = np.array([1 / (s + 1) for _, s in base])
    pesos = pesos / pesos.sum()
    vetor_medio = np.average(vetores, axis=0, weights=pesos)
    vetor_medio = vetor_medio / np.linalg.norm(vetor_medio)

    sims = M @ vetor_medio
    top = _top_n(sims, n_sugestoes)
    print(f"[centroide] sugestoes com base em: {base}")
    return [cands[i] for i in top]


def _sugerir_por_consistencia(cands, M):
    """Main phase: rank candidates by how well they explain the ranks seen so far."""
    palavras_chute = list(memory.keys())
    V = np.array([get_embedding(p) for p in palavras_chute])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    ranks = np.array([memory[p] for p in palavras_chute], dtype=np.float64)

    cons = score_consistencia(M, V, ranks)

    # once we are in the right neighborhood consistency saturates near 1, so
    # proximity to the best guess weighs more as the best rank drops
    melhor_palavra = min(memory, key=memory.get)
    melhor_rank = memory[melhor_palavra]
    if melhor_rank <= 20:
        beta = 0.6
    elif melhor_rank <= 100:
        beta = 0.4
    else:
        beta = 0.15

    v_melhor = get_embedding(melhor_palavra)
    v_melhor = v_melhor / np.linalg.norm(v_melhor)
    prox = M @ v_melhor

    final = (1 - beta) * _normalizar_01(cons) + beta * _normalizar_01(prox)

    top = _top_n(final, N_SUGESTOES_CONSISTENCIA)
    print(
        f"[consistencia] {len(memory)} chutes | melhor: {melhor_palavra}={melhor_rank} | beta={beta}"
    )
    for i in top:
        print(f"    {cands[i]:<18} consist={cons[i]:+.3f}  final={final[i]:.3f}")
    return [cands[i] for i in top]


def sugerir_proximo():
    global ultima_estrategia

    if not memory:
        ultima_estrategia = "tema"
        if temas_exploratorios:
            sugestao = random.choice(temas_exploratorios)
            temas_exploratorios.remove(sugestao)
        else:
            sugestao = random.choice(
                [
                    c
                    for c in candidatos_lista
                    if c not in palavras_usadas and c not in palavras_invalidas
                ]
            )
        _marcar(sugestao)
        return [sugestao]

    # everything is bad: no reliable direction yet, try another semantic area
    if all(s >= 500 for s in memory.values()) and len(memory) < MIN_CHUTES_CONSISTENCIA:
        if temas_exploratorios:
            ultima_estrategia = "tema"
            sugestao = random.choice(temas_exploratorios)
            temas_exploratorios.remove(sugestao)
            _marcar(sugestao)
            print(f"[explorando novo tema: {sugestao}]")
            return [sugestao]

        ultima_estrategia = "aleatorio"
        candidato = random.choice(
            [
                c
                for c in candidatos_lista
                if c not in palavras_usadas and c not in palavras_invalidas
            ]
        )
        _marcar(candidato)
        return [candidato]

    mask = np.array(
        [
            c not in memory
            and c not in palavras_usadas
            and c not in palavras_invalidas
            and candidatos_stems[i] not in stems_usados
            for i, c in enumerate(candidatos_lista)
        ]
    )
    if not mask.any():
        print("[sem candidatos restantes]")
        return []

    candidatos_filtrados = [
        candidatos_lista[i] for i in range(len(candidatos_lista)) if mask[i]
    ]
    matriz_filtrada = matriz_norm[mask]

    if len(memory) >= MIN_CHUTES_CONSISTENCIA:
        ultima_estrategia = "consistencia"
        lista_palavras = _sugerir_por_consistencia(
            candidatos_filtrados, matriz_filtrada
        )
    else:
        ultima_estrategia = "centroide"
        lista_palavras = _sugerir_por_centroide(candidatos_filtrados, matriz_filtrada)

    lista_palavras = list(dict.fromkeys(lista_palavras))
    for w in lista_palavras:
        _marcar(w)
    return lista_palavras


def enviar_chute_site(palavra):
    palavra = str(palavra)

    try:
        linha_anterior = driver.find_element(
            By.CSS_SELECTOR, ".row-wrapper.current .row"
        ).text
    except:
        linha_anterior = ""

    # um aviso remanescente do chute anterior seria lido como resposta deste;
    # espera ele sumir antes de enviar (se nao sumir, seguimos mesmo assim)
    try:
        WebDriverWait(driver, 2, poll_frequency=0.1).until(
            lambda d: not d.find_elements(By.CSS_SELECTOR, ".message-text")
        )
    except Exception:
        pass

    input_box = driver.find_element(By.CSS_SELECTOR, "input.word")
    input_box.clear()
    input_box.send_keys(palavra)
    input_box.send_keys(Keys.ENTER)

    try:

        def _esperar(d):
            rows = d.find_elements(By.CSS_SELECTOR, ".row-wrapper.current .row")
            if rows and rows[0].text != linha_anterior and rows[0].text != "":
                return True
            msgs = d.find_elements(By.CSS_SELECTOR, ".message-text")
            return bool(msgs) and _classificar_mensagem(msgs[0].text) is not None

        WebDriverWait(driver, 3, poll_frequency=0.1).until(_esperar)

        # a linha tem prioridade: se o chute valeu, ela mudou e o aviso e ruido
        rows = driver.find_elements(By.CSS_SELECTOR, ".row-wrapper.current .row")
        texto = rows[0].text.strip() if rows else ""
        if texto and texto != linha_anterior:
            palavra_retornada, score = texto.split()
            return palavra_retornada.lower(), int(score)

        msg_elements = driver.find_elements(By.CSS_SELECTOR, ".message-text")
        tipo = _classificar_mensagem(msg_elements[0].text) if msg_elements else None

        if tipo == "repetida":
            jogo_atual["repetidos"] += 1
            _marcar(palavra)
            print("Repetida:", palavra)
            return palavra, None

        if tipo in ("desconhecida", "comum"):
            registrar_palavra_invalida(palavra, tipo)
            return palavra, None

        raise RuntimeError("resposta do site nao reconhecida")
    except Exception as e:
        print(f"[erro ao ler resultado de '{palavra}': {e}]")
        return palavra, None


def reenviar_chutes_para_interface():
    print("Recarregando chutes anteriores...")
    for palavra in list(memory.keys()):
        enviar_chute_site(palavra)


def adicionar_chute_site(palavra):
    chute, score = enviar_chute_site(palavra)

    if score is None:
        return

    memory[chute] = score
    _marcar(chute)
    get_embedding(chute)
    print(f"{chute} → {score}")


def loop_automatico():
    global n_sugestoes
    reenviar_chutes_para_interface()
    while True:
        perfeitos_antes = sum(1 for s in memory.values() if s < 50)
        usando_perfeitos_puro = perfeitos_antes >= 5

        proximas = sugerir_proximo()
        if not proximas:
            print("[nenhuma sugestao possivel — encerrando]")
            return
        for i in range(len(proximas)):
            palavra = proximas[i]
            chute, score = enviar_chute_site(palavra)
            if score is not None:
                adicionar_chute(chute, score)
                dashboard.registrar_chute(jogo_atual, chute, score, ultima_estrategia)
                dashboard.salvar_historico(historico)
                dashboard.atualizar_grafico(historico, jogo_atual)
                print(f"{i+1}/{len(proximas)}:\t\t {chute} -> {score}")
                if score <= 5:
                    print(f"[top 5!]")
                if score == 1:
                    print(f"Palavra encontrada: {chute}")
                    return

        perfeitos_depois = sum(1 for s in memory.values() if s < 50)
        if usando_perfeitos_puro and perfeitos_depois == perfeitos_antes:
            n_sugestoes += 5
            print(
                f"[sem melhoria em perfeitos — aumentando sugestões para {n_sugestoes}]"
            )
        elif perfeitos_depois > perfeitos_antes:
            n_sugestoes = 10


try:
    loop_automatico()

except NoSuchElementException:
    clear_output(wait=False)
    print("\n[ERRO] Não foi possível encontrar o campo de input do jogo.")
    print(
        "\n→ Entre na partida manualmente no navegador (onde aparece o campo de digitar palavra)"
    )
    print("→ Depois execute o script novamente.\n")
