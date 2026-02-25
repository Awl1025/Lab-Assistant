# Lab-Assistant

## Project Goal
Lab assistant tool that uses a large dataset of laboratory data and research papers to create a system that ensures lab practices align with established research to
verify results. It will be an acting proofreader of lab imagery to ensure work adheres to safe and practical research procedures and other work within the lab.

### Provider Choice: 

**Generation:** Google Gemini (model="gemini-2.5-flash", temperature=X)

**Embeddings:** HuggingFace (model="sentence-transformers/all-MiniLM-L6-v2")

**Vision (if needed):** Google Gemini (model="gemini-2.5-flash")

**Why:** There are a couple benifits to going with Gemini but the most appealing seems to be the fact that it can handle all our needs with a single API. It makes the process simpler since we don't need to mess around with multiple API keys and rate limits. Another big advantage is thet it's multimodal. This will allow us the flexibility to integrate computer vision into the program for analysis of visual lab data and automate parts of analysis. It also offers 1 million tokens per day on the free tier which should be more than enough for our needs. Native multi-modality. It can process the Broad Bioimage sets and the Pes2oX text in the same context window without needing separate vision encoders like CLIP.

### Setup
Use pip install -r requirements.txt 
store the GOOGLE_API_KEY in GitHub Codespace Secrets.
