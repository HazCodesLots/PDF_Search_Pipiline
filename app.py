{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c1c489fe-9c9b-4fd6-ab5e-8e5427d47f51",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Dev\\Envs\\core_env\\Lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pdfplumber'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[3]\u001b[39m\u001b[32m, line 2\u001b[39m\n\u001b[32m      1\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mgradio\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mgr\u001b[39;00m\n\u001b[32m----> \u001b[39m\u001b[32m2\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mpdfplumber\u001b[39;00m\n\u001b[32m      3\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mfaiss\u001b[39;00m\n\u001b[32m      4\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnumpy\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnp\u001b[39;00m\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'pdfplumber'"
     ]
    }
   ],
   "source": [
    "import gradio as gr\n",
    "import pdfplumber\n",
    "import faiss\n",
    "import numpy as np\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "import torch\n",
    "\n",
    "\n",
    "embedder = SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\")\n",
    "qna_tokenizer = AutoTokenizer.from_pretrained(\"microsoft/phi-2\")\n",
    "qna_model = AutoModelForCausalLM.from_pretrained(\"microsoft/phi-2\")\n",
    "\n",
    "\n",
    "text_chunks = []\n",
    "metadata = []\n",
    "index = None\n",
    "\n",
    "\n",
    "def extract_text_from_pdf(pdf_file):\n",
    "    global text_chunks, metadata, index\n",
    "    text_chunks = []\n",
    "    metadata = []\n",
    "\n",
    "    with pdfplumber.open(pdf_file.name) as pdf:\n",
    "        for i, page in enumerate(pdf.pages):\n",
    "            text = page.extract_text()\n",
    "            if text:\n",
    "                text_chunks.append(text)\n",
    "                metadata.append(f\"Page {i+1}\")\n",
    "\n",
    "    # Embed text\n",
    "    if text_chunks:\n",
    "        embeddings = embedder.encode(text_chunks, convert_to_numpy=True)\n",
    "        index = faiss.IndexFlatL2(embeddings.shape[1])\n",
    "        index.add(embeddings)\n",
    "        return f\"✅ Indexed {len(text_chunks)} text chunks.\"\n",
    "    else:\n",
    "        return \"⚠️ No text found in the PDF.\"\n",
    "\n",
    "# Function: Search and ask follow-up\n",
    "def search_and_answer(query, top_k=3, follow_up=\"What is this text about?\"):\n",
    "    if index is None:\n",
    "        return \"⚠️ Please upload and process a PDF first.\", \"\"\n",
    "\n",
    "    # Embed query\n",
    "    query_emb = embedder.encode([query])\n",
    "    D, I = index.search(np.array(query_emb), top_k)\n",
    "    results = [text_chunks[i] for i in I[0]]\n",
    "    result_text = \"\"\n",
    "\n",
    "    for i, chunk in enumerate(results):\n",
    "        result_text += f\"📄 Result {i+1} (from {metadata[I[0][i]]}):\\n{chunk.strip()}\\n\\n\"\n",
    "\n",
    "    # Run Q&A on first result\n",
    "    prompt = f\"{results[0]}\\n\\nQ: {follow_up}\\nA:\"\n",
    "    input_ids = qna_tokenizer(prompt, return_tensors=\"pt\").input_ids\n",
    "    output = qna_model.generate(input_ids, max_new_tokens=100, do_sample=False)\n",
    "    answer = qna_tokenizer.decode(output[0], skip_special_tokens=True).split(\"A:\")[-1].strip()\n",
    "\n",
    "    return result_text, answer\n",
    "\n",
    "# Gradio UI\n",
    "with gr.Blocks() as demo:\n",
    "    gr.Markdown(\"# 📘 PDF Semantic Search + Q&A (CPU-friendly)\")\n",
    "\n",
    "    with gr.Row():\n",
    "        with gr.Column():\n",
    "            pdf_input = gr.File(label=\"Upload PDF\")\n",
    "            upload_btn = gr.Button(\"📥 Process PDF\")\n",
    "            upload_output = gr.Textbox(label=\"Status\")\n",
    "\n",
    "        with gr.Column():\n",
    "            query_input = gr.Textbox(label=\"🔍 Search Query\", placeholder=\"e.g. spectrogram diagram\")\n",
    "            followup_input = gr.Textbox(label=\"💬 Follow-up Question\", value=\"What is this text about?\")\n",
    "            search_btn = gr.Button(\"🔎 Search and Ask\")\n",
    "    \n",
    "    results_text = gr.Textbox(label=\"📄 Search Results\", lines=10)\n",
    "    answer_text = gr.Textbox(label=\"🤖 Answer\", lines=4)\n",
    "\n",
    "    upload_btn.click(fn=extract_text_from_pdf, inputs=pdf_input, outputs=upload_output)\n",
    "    search_btn.click(fn=search_and_answer, inputs=[query_input, followup_input], outputs=[results_text, answer_text])\n",
    "\n",
    "demo.launch()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9c6d3138-d625-494a-825b-76889dafdb4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"requirements.txt\", \"w\") as f:\n",
    "    f.write(\"\"\"gradio\n",
    "pdfplumber\n",
    "faiss-cpu\n",
    "sentence-transformers\n",
    "transformers\n",
    "torch\n",
    "\"\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c7d7140-f7d3-46db-8ad7-6e9aca5b3dce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (core_env)",
   "language": "python",
   "name": "core_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
