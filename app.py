import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()


SYSTEM_PROMPT = """
You are a comparative scripture assistant.

Rules:
1. Use ONLY the retrieved text.
2. Never merge teachings.
3. Answer separately for each scripture.
4. If a scripture does not address the question, say:
   \"No direct reference found.\"

Format EXACTLY like this:

Gita says:
- ...

Bible says:
- ...

Quran says:
- ...
""".strip()

# Council configuration
COUNCIL_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile", 
    "moonshotai/kimi-k2-instruct-0905"
]

# Model to book assignments
MODEL_BOOK_ASSIGNMENTS = {
    "openai/gpt-oss-120b": "gita",
    "llama-3.3-70b-versatile": "bible", 
    "moonshotai/kimi-k2-instruct-0905": "quran"
}

CHAIRMAN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def query_groq_model(model: str, messages: List[Dict[str, str]], timeout: float = 120.0, max_tokens: int = 150) -> Optional[Dict[str, Any]]:
    """Query a single Groq model."""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return {"content": data['choices'][0]['message'].get('content')}
    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_groq_models_parallel(models: List[str], messages: List[Dict[str, str]], max_tokens: int = 150) -> Dict[str, Optional[Dict[str, Any]]]:
    """Query multiple Groq models in parallel."""
    tasks = [query_groq_model(model, messages, max_tokens=max_tokens) for model in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """Parse the FINAL RANKING section from the model's response."""
    import re
    
    if "FINAL RANKING:" in ranking_text:
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches
    
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(stage2_results: List[Dict[str, Any]], label_to_model: Dict[str, str]) -> List[Dict[str, Any]]:
    """Calculate aggregate rankings across all models."""
    from collections import defaultdict
    
    model_positions = defaultdict(list)
    
    for ranking in stage2_results:
        parsed_ranking = parse_ranking_from_text(ranking['ranking'])
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
    
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })
    
    aggregate.sort(key=lambda x: x['average_rank'])
    return aggregate


def get_book_context(question: str, book: str, retriever) -> str:
    """Get relevant context from a specific religious book."""
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever.invoke(question)
    
    # Filter documents for the specific book
    book_docs = [doc for doc in docs if doc.metadata.get('source') == book]
    
    if not book_docs:
        return f"No relevant passages found in {book.title()} for this question."
    
    context = "\n\n".join([
        f"[{doc.metadata.get('source', '').upper()}]\n{doc.page_content}" 
        for doc in book_docs[:5]  # Limit to top 5 most relevant passages
    ])
    
    return context
    """Calculate aggregate rankings across all models."""
    from collections import defaultdict
    
    model_positions = defaultdict(list)
    
    for ranking in stage2_results:
        parsed_ranking = parse_ranking_from_text(ranking['ranking'])
        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)
    
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })
    
    aggregate.sort(key=lambda x: x['average_rank'])
    return aggregate


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_or_load_vectorstore() -> FAISS:
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_dir = "scripture_faiss"

    if os.path.isdir(index_dir):
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs: list[Document] = []

    for source in ["gita", "bible", "quran"]:
        txt_path = os.path.join("data", source, f"{source}.txt")
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(
                f"Missing {txt_path}. Run the notebook to generate the .txt files first."
            )
        text = _load_text(txt_path)
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"source": source}))

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_dir)
    return db


@st.cache_resource
def initialize() -> tuple[FAISS, ChatGroq]:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment or a .env file."
        )

    db = _build_or_load_vectorstore()
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return db, llm


def ask(question: str, retriever, llm: ChatGroq) -> str:
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(question)
    else:
        docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[{d.metadata.get('source', '').upper()}]\n{d.page_content}" for d in docs
    )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}
"""

    result = llm.invoke(prompt).content
    if isinstance(result, str):
        return result
    return "\n".join(str(part) for part in result)


async def run_council_debate_with_display(question: str, retriever):
    """Run the LLM council debate where each model defends their assigned religious book."""
    
    try:
        st.write(f"### 🥊 Council Debate: {question}")
        st.write("Each model will defend their assigned religious book using relevant passages!")
        
        # Show model assignments
        st.write("**📚 Model Assignments:**")
        for model, book in MODEL_BOOK_ASSIGNMENTS.items():
            st.write(f"• **{model}** defends **{book.title()}**")
        
        st.write("---")
        
        # Stage 1: Each model defends their assigned book
        st.write("### 🥊 Stage 1: Models Defending Their Books")
        st.write("Each model is retrieving relevant passages and forming their defense...")
        
        stage1_results = []
        model_containers = {}
        
        # Create containers for each model
        for model in COUNCIL_MODELS:
            assigned_book = MODEL_BOOK_ASSIGNMENTS[model]
            with st.expander(f"🤖 {model} defending {assigned_book.title()}...", expanded=True):
                model_containers[model] = st.empty()
                model_containers[model].write(f"🔍 Searching {assigned_book.title()} for relevant passages...")
        
        # Query all models in parallel
        defense_tasks = []
        for model in COUNCIL_MODELS:
            assigned_book = MODEL_BOOK_ASSIGNMENTS[model]
            book_context = get_book_context(question, assigned_book, retriever)
            
            defense_prompt = f"""You are defending the {assigned_book.title()} in a debate. Your task is to provide the best answer to the following question using ONLY the {assigned_book.title()}'s teachings.

Question: {question}

Relevant passages from {assigned_book.title()}:
{book_context}

Your response should:
1. Directly answer the question using the {assigned_book.title()}'s wisdom
2. Quote specific passages when relevant
3. Explain why the {assigned_book.title()}'s approach is the most effective
4. Be persuasive and well-reasoned
5. Stay focused on your assigned book - do not mention other religious texts

IMPORTANT: Keep your response to exactly 100 words or less. Be concise and impactful.

Defend the {assigned_book.title()}'s teachings as the best solution to this question:"""

            messages = [{"role": "user", "content": defense_prompt}]
            defense_tasks.append((model, query_groq_model(model, messages, max_tokens=120)))  # 100 words + buffer
        
        # Wait for all defenses
        for model, task in defense_tasks:
            response = await task
            assigned_book = MODEL_BOOK_ASSIGNMENTS[model]
            
            if response is not None:
                defense_text = response.get('content', '')
                model_containers[model].markdown(f"**🛡️ {assigned_book.title()} Defense:**\n\n{defense_text}")
                stage1_results.append({
                    "model": model,
                    "book": assigned_book,
                    "response": defense_text
                })
            else:
                model_containers[model].error(f"❌ {model} failed to defend {assigned_book.title()}")
        
        if not stage1_results:
            st.error("All models failed to provide defenses")
            return
        
        st.success(f"✅ Stage 1 Complete! {len(stage1_results)} models have defended their books.")
        
        # Stage 2: Cross-examination and ranking
        st.write("### 🏆 Stage 2: Cross-Examination & Ranking")
        st.write("Now each model will critique the other defenses and rank them...")
        
        labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...
        label_to_model = {f"Response {label}": f"{result['model']} (defending {result['book'].title()})" for label, result in zip(labels, stage1_results)}
        
        # Show the defenses that models will judge
        st.write("**📖 Defenses to be judged:**")
        for label, result in zip(labels, stage1_results):
            with st.expander(f"Response {label}: {result['book'].title()} defense", expanded=False):
                st.write(result['response'])
        
        responses_text = "\n\n".join([
            f"Response {label} (defending {result['book'].title()}):\n{result['response']}"
            for label, result in zip(labels, stage1_results)
        ])
        
        ranking_prompt = f"""You are judging a religious debate where each model defended their assigned book in response to: "{question}"

Here are the defenses (anonymized):

{responses_text}

Your task:
1. Evaluate each defense on:
   - How well it answers the question
   - Quality of scriptural evidence provided
   - Persuasiveness of the argument
   - Depth of understanding shown
2. Provide brief critique of each response (keep critiques concise)
3. Then provide your final ranking

IMPORTANT: 
- Keep your entire response to 150 words or less
- Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")

Now provide your evaluation and ranking:"""

        ranking_messages = [{"role": "user", "content": ranking_prompt}]
        
        # Create placeholders for ranking thinking
        ranking_containers = {}
        for model in COUNCIL_MODELS:
            assigned_book = MODEL_BOOK_ASSIGNMENTS[model]
            with st.expander(f"⚖️ {model} ({assigned_book.title()}) judging other defenses...", expanded=True):
                ranking_containers[model] = st.empty()
                ranking_containers[model].write("🤔 Analyzing each defense carefully...")
        
        ranking_responses = await query_groq_models_parallel(COUNCIL_MODELS, ranking_messages, max_tokens=200)  # 150 words + buffer
        
        stage2_results = []
        for model, response in ranking_responses.items():
            assigned_book = MODEL_BOOK_ASSIGNMENTS[model]
            if response is not None:
                full_text = response.get('content', '')
                parsed = parse_ranking_from_text(full_text)
                
                # Update container with the ranking analysis
                ranking_containers[model].markdown(f"**📊 Cross-Examination by {assigned_book.title()} defender:**\n\n{full_text}")
                
                stage2_results.append({
                    "model": model,
                    "book": assigned_book,
                    "ranking": full_text,
                    "parsed_ranking": parsed
                })
            else:
                ranking_containers[model].error(f"❌ {model} failed to provide rankings")
        
        # Calculate aggregate rankings
        aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
        
        st.success(f"✅ Stage 2 Complete! Models have cross-examined each other's defenses.")
        
        if aggregate_rankings:
            st.write("**📊 Final Rankings:**")
            for i, rank in enumerate(aggregate_rankings, 1):
                # Extract book from the model name
                model_name = rank['model'].split(' (defending ')[0] if ' (defending ' in rank['model'] else rank['model']
                book_name = MODEL_BOOK_ASSIGNMENTS.get(model_name, 'Unknown')
                st.write(f"{i}. **{book_name.title()}** (defended by {model_name}) - Average rank: {rank['average_rank']} ({rank['rankings_count']} votes)")
        
        # Stage 3: Chairman's final verdict
        st.write("### 🎯 Stage 3: Chairman's Final Verdict")
        st.write(f"The Chairman ({CHAIRMAN_MODEL}) will now synthesize all arguments and declare the winner...")
        
        stage1_text = "\n\n".join([
            f"Model: {result['model']} (defending {result['book'].title()})\nDefense: {result['response']}"
            for result in stage1_results
        ])
        
        stage2_text = "\n\n".join([
            f"Judge: {result['model']} (defending {result['book'].title()})\nCross-examination: {result['ranking']}"
            for result in stage2_results
        ])
        
        chairman_prompt = f"""You are the Chairman of a Religious Council Debate. The question was: "{question}"

Each model defended their assigned religious book using relevant passages and teachings.

STAGE 1 - Book Defenses:
{stage1_text}

STAGE 2 - Cross-Examinations:
{stage2_text}

Your task as Chairman is to:
1. Analyze which book provided the most compelling answer to the question
2. Consider the quality of scriptural evidence presented
3. Evaluate the persuasiveness of each argument
4. DECLARE ONE CLEAR WINNER - you must choose exactly ONE book as the best answer
5. Explain specifically why that book's approach was superior to the others

CRITICAL: You MUST declare a single winning book. Do not give a balanced answer or say "all books are good." Pick the ONE book that best answered: "{question}"

Format your response as:
**WINNER: [Book Name]**
[Explanation of why this book won - max 150 words]

Provide your final verdict on which ONE religious book offers the best guidance for: "{question}":"""

        chairman_messages = [{"role": "user", "content": chairman_prompt}]
        
        # Show chairman thinking
        with st.expander(f"👑 Chairman {CHAIRMAN_MODEL} making final verdict...", expanded=True):
            chairman_container = st.empty()
            chairman_container.write("🧠 Analyzing all defenses and cross-examinations to reach final verdict...")
            
            chairman_response = await query_groq_model(CHAIRMAN_MODEL, chairman_messages, max_tokens=250)  # 200 words + buffer
            
            if chairman_response is None:
                chairman_container.error("❌ Chairman failed to provide final verdict")
            else:
                final_verdict = chairman_response.get('content', '')
                
                # Extract winner from the response if formatted correctly
                winner_book = "Unknown"
                if "**WINNER:" in final_verdict:
                    try:
                        winner_line = final_verdict.split("**WINNER:")[1].split("**")[0].strip()
                        winner_book = winner_line
                    except:
                        pass
                
                # Display with emphasis on winner
                chairman_container.markdown(f"**👑 Chairman's Final Verdict:**\n\n{final_verdict}")
                
                # Show prominent winner announcement
                if winner_book != "Unknown":
                    st.success(f"🏆 **DEBATE WINNER: {winner_book.upper()}** 🏆")
                else:
                    st.info("🏆 **DEBATE COMPLETE** - See chairman's verdict above")
        
        st.balloons()  # Celebration for the winner!
        st.success("🏆 Council Debate Complete! A winner has been declared!")
        
        # Store results in session state
        st.session_state.council_result = {
            "question": question,
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": {"model": CHAIRMAN_MODEL, "response": chairman_response.get('content', '') if chairman_response else "Error"},
            "metadata": {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings,
                "model_assignments": MODEL_BOOK_ASSIGNMENTS
            },
            "success": True
        }
        
    except Exception as e:
        st.error(f"💥 Council debate crashed: {str(e)}")


async def run_council_debate_with_display_old():
    """Run the LLM council debate and display results in main screen."""
    debate_question = """Which religious book (Gita, Bible, or Quran) has the best rules to follow for life? 
    
    Consider factors like:
    - Practical guidance for daily living
    - Moral and ethical framework
    - Wisdom for handling life's challenges
    - Universal applicability across cultures and times
    - Balance between spiritual and worldly matters
    
    Provide a well-reasoned argument for your choice, acknowledging the strengths of other texts while making your case."""
    
    try:
        # Stage 1: Collect individual responses
        st.write("### 🥊 Stage 1: Models Fighting - Individual Opinions")
        st.write("Each model is now forming their opinion on which religious book has the best life rules...")
        
        messages = [{"role": "user", "content": debate_question}]
        
        # Create placeholders for each model's thinking
        model_containers = {}
        for model in COUNCIL_MODELS:
            with st.expander(f"🤖 {model} is thinking...", expanded=True):
                model_containers[model] = st.empty()
                model_containers[model].write("🧠 Analyzing religious texts...")
        
        responses = await query_groq_models_parallel(COUNCIL_MODELS, messages)
        
        stage1_results = []
        for model, response in responses.items():
            if response is not None:
                # Update the container with the actual response
                model_containers[model].markdown(f"**Final Opinion:**\n\n{response.get('content', '')}")
                stage1_results.append({
                    "model": model,
                    "response": response.get('content', '')
                })
            else:
                model_containers[model].error(f"❌ {model} failed to respond")
        
        if not stage1_results:
            st.error("All models failed to respond")
            return
        
        st.success(f"✅ Stage 1 Complete! {len(stage1_results)} models have shared their opinions.")
        
        # Stage 2: Collect rankings
        st.write("### 🏆 Stage 2: Models Judging Each Other")
        st.write("Now each model will anonymously rank the other responses...")
        
        labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...
        label_to_model = {f"Response {label}": result['model'] for label, result in zip(labels, stage1_results)}
        
        # Show the anonymized responses that models will judge
        st.write("**Anonymized responses for judging:**")
        for label, result in zip(labels, stage1_results):
            with st.expander(f"Response {label} (from {result['model']})", expanded=False):
                st.write(result['response'])
        
        responses_text = "\n\n".join([
            f"Response {label}:\n{result['response']}"
            for label, result in zip(labels, stage1_results)
        ])
        
        ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {debate_question}

Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

        ranking_messages = [{"role": "user", "content": ranking_prompt}]
        
        # Create placeholders for ranking thinking
        ranking_containers = {}
        for model in COUNCIL_MODELS:
            with st.expander(f"⚖️ {model} is judging responses...", expanded=True):
                ranking_containers[model] = st.empty()
                ranking_containers[model].write("🤔 Evaluating each response carefully...")
        
        ranking_responses = await query_groq_models_parallel(COUNCIL_MODELS, ranking_messages)
        
        stage2_results = []
        for model, response in ranking_responses.items():
            if response is not None:
                full_text = response.get('content', '')
                parsed = parse_ranking_from_text(full_text)
                
                # Update container with the ranking analysis
                ranking_containers[model].markdown(f"**Ranking Analysis:**\n\n{full_text}")
                
                stage2_results.append({
                    "model": model,
                    "ranking": full_text,
                    "parsed_ranking": parsed
                })
            else:
                ranking_containers[model].error(f"❌ {model} failed to provide rankings")
        
        # Calculate aggregate rankings
        aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)
        
        st.success(f"✅ Stage 2 Complete! Models have ranked each other's responses.")
        
        if aggregate_rankings:
            st.write("**📊 Aggregate Rankings:**")
            for i, rank in enumerate(aggregate_rankings, 1):
                st.write(f"{i}. **{rank['model']}** - Average rank: {rank['average_rank']} ({rank['rankings_count']} votes)")
        
        # Stage 3: Chairman synthesis
        st.write("### 🎯 Stage 3: Chairman's Final Decision")
        st.write(f"The Chairman ({CHAIRMAN_MODEL}) is now synthesizing all opinions into a final verdict...")
        
        stage1_text = "\n\n".join([
            f"Model: {result['model']}\nResponse: {result['response']}"
            for result in stage1_results
        ])
        
        stage2_text = "\n\n".join([
            f"Model: {result['model']}\nRanking: {result['ranking']}"
            for result in stage2_results
        ])
        
        chairman_prompt = f"""You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {debate_question}

STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

        chairman_messages = [{"role": "user", "content": chairman_prompt}]
        
        # Show chairman thinking
        with st.expander(f"👑 Chairman {CHAIRMAN_MODEL} is making final decision...", expanded=True):
            chairman_container = st.empty()
            chairman_container.write("🧠 Analyzing all opinions and rankings to reach final verdict...")
            
            chairman_response = await query_groq_model(CHAIRMAN_MODEL, chairman_messages)
            
            if chairman_response is None:
                chairman_container.error("❌ Chairman failed to provide final decision")
            else:
                final_decision = chairman_response.get('content', '')
                chairman_container.markdown(f"**👑 Chairman's Final Verdict:**\n\n{final_decision}")
        
        st.success("🏆 Council Debate Complete! The verdict is in!")
        
        # Store results in session state for potential later reference
        st.session_state.council_result = {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "stage3": {"model": CHAIRMAN_MODEL, "response": chairman_response.get('content', '') if chairman_response else "Error"},
            "metadata": {
                "label_to_model": label_to_model,
                "aggregate_rankings": aggregate_rankings
            },
            "success": True
        }
        
    except Exception as e:
        st.error(f"💥 Council debate crashed: {str(e)}")


async def run_council_debate() -> dict:
    """Legacy function for backward compatibility."""
    # This is kept for any potential future use but main display now uses run_council_debate_with_display
    pass


def main() -> None:
    st.title("Comparative Scripture Assistant")
    st.caption("Gita • Bible • Quran (RAG)")

    try:
        db, llm = initialize()
    except Exception as e:
        st.error(str(e))
        st.stop()

    retriever = db.as_retriever(search_kwargs={"k": 15})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Mode Selection")
        
        # Mode selector
        mode = st.selectbox(
            "Choose Mode:",
            ["Scripture Q&A", "Council Debate"],
            help="Scripture Q&A: Ask questions about religious texts\nCouncil Debate: LLMs debate which religious book has the best life rules"
        )
        
        if mode == "Council Debate":
            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not found. Please add it to your .env file.")
                st.stop()
            
            st.info("🏛️ **Council Mode**: Ask any question and watch models defend their assigned books!")
            st.write("**📚 Book Assignments:**")
            for model, book in MODEL_BOOK_ASSIGNMENTS.items():
                st.write(f"• {book.title()}: {model}")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            if "council_result" in st.session_state:
                del st.session_state.council_result
            if "show_council_debate" in st.session_state:
                del st.session_state.show_council_debate
            st.rerun()

    # Main content area
    if mode == "Council Debate":
        # Council mode - show question input and debate
        st.write("### 🏛️ Ask the Council!")
        st.write("Enter your question and watch the models defend their assigned religious books:")
        
        # Show model assignments
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🕉️ **Gita**\nDefended by:\nllama-3.3-70b-versatile")
        with col2:
            st.info("✝️ **Bible**\nDefended by:\nllama-3.1-70b-versatile")
        with col3:
            st.info("☪️ **Quran**\nDefended by:\nmistral-8x7b-32768")
        
        # Question input for council mode
        if council_question := st.chat_input("Ask your question to the Council..."):
            # Run the council debate with the user's question
            asyncio.run(run_council_debate_with_display(council_question, retriever))
            
    elif mode == "Scripture Q&A":
        # Regular Scripture Q&A mode
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if question := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask(question, retriever=retriever, llm=llm)
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # Default welcome screen
        st.write("### Welcome to the Comparative Scripture Assistant!")
        st.write("Choose a mode from the sidebar to get started.")


if __name__ == "__main__":
    main()
