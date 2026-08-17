import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import random
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support import expected_conditions as EC
from IPython.display import clear_output
from webdriver_manager.chrome import ChromeDriverManager
import time


# Inicializa navegador
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://contexto.me/en/")

wait = WebDriverWait(driver, 20)

# Clica no link do daily game (seletor exato pelo href)
play_link = wait.until(EC.element_to_be_clickable((
    By.CSS_SELECTOR, "a[href='/en/daily']"
)))
play_link.click()

# Espera o campo de input do jogo carregar
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input.word")))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
print("BERT Carregado com sucesso!")


def encode_words(words):
	"""Embedding de cada palavra via mean pooling apenas sobre os tokens reais.

	[PAD] fica de fora (senao o vetor dependeria do tamanho da maior palavra
	do batch) e [CLS]/[SEP] tambem (num input de 1 palavra eles dominariam a
	media). Assim o vetor de uma palavra e o mesmo em batch ou sozinha.
	"""
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

	# tokens reais = atendidos pela attention e nao especiais
	pool_mask = (inputs["attention_mask"] * (1 - special_mask)).unsqueeze(-1).float()
	soma = (outputs.last_hidden_state * pool_mask).sum(dim=1)
	qtd = pool_mask.sum(dim=1).clamp(min=1e-9)
	return (soma / qtd).numpy()


# Embeddings cache
embeddings = {}
memory = {}

# Pre-computa embeddings ou carrega do arquivo se ja existir
# _v2: o arquivo antigo foi gerado com pooling sobre [PAD]/[CLS]/[SEP] e e invalido
EMBEDDINGS_FILE = "embeddingsValues_v2.npz"

if os.path.exists(EMBEDDINGS_FILE):
	print("Arquivo de embeddings encontrado.")
	data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
	candidatos_raw = data["candidatos"]
	matriz_raw     = data["matriz"]

	# Converte para str puro e deduplica preservando primeira ocorrência
	seen_load = {}
	for i, w in enumerate(candidatos_raw):
		w_str = str(w)
		if w_str not in seen_load:
			seen_load[w_str] = i
	indices_unicos    = sorted(seen_load.values())
	candidatos_lista  = [str(candidatos_raw[i]) for i in indices_unicos]
	matriz_candidatos = matriz_raw[indices_unicos]

	for word, vec in zip(candidatos_lista, matriz_candidatos):
		embeddings[word] = vec
	print(f"Pronto! {len(candidatos_lista)} candidatos carregados ({len(candidatos_raw) - len(candidatos_lista)} duplicatas removidas).")
else:
	print("Arquivo nao encontrado. Pre-computando embeddings")
	todos_candidatos = [w for w in tokenizer.vocab.keys() if w.isalpha() and len(w) > 4]

	batch_size = 64
	candidatos_lista = []
	matriz_candidatos = []

	for i in range(0, len(todos_candidatos), batch_size):
		batch = todos_candidatos[i:i + batch_size]
		embs = encode_words(batch)
		for j, word in enumerate(batch):
			candidatos_lista.append(word)
			matriz_candidatos.append(embs[j])
			embeddings[word] = embs[j]

	matriz_candidatos = np.array(matriz_candidatos)
	np.savez(EMBEDDINGS_FILE, candidatos=candidatos_lista, matriz=matriz_candidatos)
	print(f"Pronto! {len(candidatos_lista)} candidatos pre-computados e salvos em '{EMBEDDINGS_FILE}'.")

# Pre-computa radicais de todos os candidatos (feito uma vez só)
stemmer = PorterStemmer()
candidatos_stems = [stemmer.stem(w) for w in candidatos_lista]
print(f"Radicais pre-computados para {len(candidatos_stems)} candidatos.")

# Palavras genéricas para abrir sentidos semânticos
temas_exploratorios = [
	"object", "place", "person", "animal", "emotion",
	"food", "vehicle", "technology", "family", "music",
	"clothing", "plant", "profession", "color"
]

palavras_usadas = set()
stems_usados = set()
n_sugestoes = 10

def _marcar(palavra):
	palavras_usadas.add(palavra)
	stems_usados.add(stemmer.stem(palavra))

# Função para obter embedding
def get_embedding(word):
	if word in embeddings:
		return embeddings[word]

	emb = encode_words([word])[0]
	embeddings[word] = emb
	return emb

# Adiciona chute manual
def adicionar_chute(palavra, score):
	memory[palavra] = score
	_marcar(palavra)
	get_embedding(palavra)

# Sugestão adaptativa
def sugerir_proximo():
	if not memory:
		sugestao = random.choice(temas_exploratorios)
		temas_exploratorios.remove(sugestao)
		_marcar(sugestao)
		return [sugestao]

	# Classificação refinada por score
	perfeitos = sorted([(p, s) for p, s in memory.items() if s < 50],         key=lambda x: x[1])
	otimos    = sorted([(p, s) for p, s in memory.items() if 50  <= s < 200], key=lambda x: x[1])
	ruins     = [(p, s) for p, s in memory.items() if s >= 500]

	if len(ruins) == len(memory):
		if temas_exploratorios:
			sugestao = random.choice(temas_exploratorios)
			temas_exploratorios.remove(sugestao)
			_marcar(sugestao)
			print(f"[explorando novo tema: {sugestao}]")
			return [sugestao]
		else:
			candidato = random.choice([c for c in candidatos_lista if c not in palavras_usadas])
			_marcar(candidato)
			return [candidato]

	MIN_BASE = 10
	base = []
	for group in [perfeitos, otimos]:
		if len(base) >= MIN_BASE:
			break
		base.extend(list(group)[:MIN_BASE - len(base)])

	if not base:
		base = sorted(memory.items(), key=lambda x: x[1])[:MIN_BASE]

	vetores = np.array([get_embedding(p) for p, _ in base])
	pesos = np.array([1 / (s + 1) for _, s in base])
	pesos = pesos / pesos.sum()
	vetor_medio = np.average(vetores, axis=0, weights=pesos)

	mask = np.array([
		c not in memory and c not in palavras_usadas and candidatos_stems[i] not in stems_usados
		for i, c in enumerate(candidatos_lista)
	])
	candidatos_filtrados = [candidatos_lista[i] for i in range(len(candidatos_lista)) if mask[i]]
	matriz_filtrada = matriz_candidatos[mask]

	sims = cosine_similarity([vetor_medio], matriz_filtrada)[0]
	top_indices = np.argsort(sims)[-n_sugestoes:][::-1]
	lista_palavras = list(dict.fromkeys([candidatos_filtrados[i] for i in top_indices]))
	print(f"Próximas sugestões carregadas com base em: {base}")
	for w in lista_palavras:
		_marcar(w)
	return lista_palavras

def enviar_chute_site(palavra):
	palavra = str(palavra)

	try:
		linha_anterior = driver.find_element(By.CSS_SELECTOR, ".row-wrapper.current .row").text
	except:
		linha_anterior = ""

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
			return bool(msgs) and "guessed" in msgs[0].text

		WebDriverWait(driver, 3, poll_frequency=0.1).until(_esperar)

		msg_elements = driver.find_elements(By.CSS_SELECTOR, ".message-text")
		if msg_elements and "guessed" in msg_elements[0].text:
			_marcar(palavra)
			print('Repetida:', palavra)
			return palavra, None

		resultado = driver.find_element(By.CSS_SELECTOR, ".row-wrapper.current .row")
		texto = resultado.text.strip()
		palavra_retornada, score = texto.split()
		return palavra_retornada.lower(), int(score)
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
		for i in range(len(proximas)):
			palavra = proximas[i]
			chute, score = enviar_chute_site(palavra)
			if score is not None:
				adicionar_chute(chute, score)
				print(f"{i+1}/{len(proximas)}:\t\t {chute} -> {score}")
				if score <= 5:
					print(f"[top 5!]")
				if score == 1:
					print(f"Palavra encontrada: {chute}")
					return

		perfeitos_depois = sum(1 for s in memory.values() if s < 50)
		if usando_perfeitos_puro and perfeitos_depois == perfeitos_antes:
			n_sugestoes += 5
			print(f"[sem melhoria em perfeitos — aumentando sugestões para {n_sugestoes}]")
		elif perfeitos_depois > perfeitos_antes:
			n_sugestoes = 10

# Logica para adicionar chutes manuais antes de iniciar o loop automático
# listaPalavras = []
# if len(listaPalavras) == 0:
#     print("Nenhuma palavra foi passada para o loop automático.") 
#     for palavra in listaPalavras:
#         adicionar_chute_site(palavra)
#         print(f"Enviando chute manual: {palavra}")
# else:
#     print("Iniciando jogo")
try:
	loop_automatico()

except NoSuchElementException:
	clear_output(wait=False)
	print("\n[ERRO] Não foi possível encontrar o campo de input do jogo.")
	print("\n→ Entre na partida manualmente no navegador (onde aparece o campo de digitar palavra)")
	print("→ Depois execute o script novamente.\n")
