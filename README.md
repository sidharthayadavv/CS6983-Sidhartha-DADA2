# DADA2 — Agentic RAG over SEC filings

**Fill in both sections at the bottom before you submit.** Keep the deployed
app line exactly in this form; it is how your app gets found and graded.

## Getting started

There is no separate spec. The instructions are in the notebooks, and
`DADA2_00_Start_Here.ipynb` is the one to open first: it explains the
assignment, the setup, the grading, and what each of the four notebooks after
it asks for. Work through them in order.

Get the files the same way as every class day: pull, then work in a copy:

```bash
cd lecture-files
git pull
cd ..
cp -r lecture-files/DADA2 DADA2     # work in the copy, not in lecture-files/
cd DADA2
cp example.env .env                  # then fill in your keys
```

Then open `DADA2_00_Start_Here.ipynb` from your `DADA2` copy and run its
setup check.

Deploying is `DEPLOY.md`.

---

## Submission

```
Deployed app: https://sec-filings-agent-8eh6.onrender.com
```

ADD YOUR NAME TO YOUR RENDER PROJECT SO THE LINK CAN BE PERSONALIZED.

PLEASE ENSURE YOUR URL WORKS BY TESTING IT IN INCOGNITO MODE.

### Write-up

Paste the write-up from notebook 04 (TODO 4.3) here, under the same five
headings.

**Conditions.**

Chat model gemini-3.5-flash-lite.
Decision model clef-flash on Cloudflare Workers AI.
Reranker Cohere rerank-v3.5.
Embeddings qwen3-embedding-0.6b through Cloudflare, stored in Pinecone. Chunks of 1,200 characters with 150 overlap, k = 5, a rerank pool of 10, and RRF constant 60. 
The agent used MAX_ATTEMPTS = 3, SURE = 0.6 and ABSENT_SURE = 0.8. 
The routers were scored on all 24 labeled questions (14 specific, 10 comparative). The agents were scored on 14 (the first 8 specific and the first 6 comparative), and the same chat model judged every answer.

**The routers.**

The 2 routers decision model and chat model tied on accuracy with 20/24.
As they tied on accuracy, and as decision model was faster by ~30%(around 0.23s faster) choosing decision model is a better choice (decision model: 0.53s vs chat model: 0.76s). Also, decision model returns probability for every option and we can check how sure the model is and can take actions. If it’s below SURE the router falls back to default strategy. Also, decision calls don’t count against the 500-a-day gemini limit. So every routing and diagnosis call it makes is a chat call saved.

**The agents.**

Ladder scored an accuracy of 0.79 with 4.7 chat calls and 16.7s per question. Whereas, diagnosis with 0.71 accuracy, 2.6 chat calls, 2.3 decision calls and 9s per question (0.79 is 11/14 and 0.71 is 10/14, diagnosis lost like one question). Diagnosis used fewer chat calls around 45% and 46% less time per question. They differed most on “What financial charge did NVIDIA incur in fiscal year 2026”, which the ladder answered correctly and diagnosis got wrong. Diagnosis routed it correctly to the filter, so it was not a routing failure. After one search the decision model said sufficient with only 0.35 from {"sufficient": 0.35, "incomplete":0.30, "wrong_words": 0.25}. decide stops on sufficient at any probability, so the agent answered from the first five passages,even though the model put 0.55 on the passages not being enough. Both agents start with the same filter search, so the difference was the judgment step. A confidence floor on sufficient, like the one absent already has with ABSENT_SURE, would have sent it to a stronger search.

**What did not help.**

The reranker did not help. In notebook 2 with 6 question block, both filter+rrf and filter+rrf+rerank scored a recall of 1.00 and MRR of 0.806, so adding ranker changed nothing. The reason is the pipeline , not the model. Both metrics are measured at the file level, and once the filter restricts the search to one ticker and fiscal year, the pool can only come from three section files of a single filing. Recall was already 1.00, and with so few files to choose between, reranking didn’t change which file came first on any of the six questions, so MRR stayed at 0.806.
The reranker does help without a filter. Using base+reranker gave us 0.83 recall against 0.67 and MRR changed from 0.325 to 0.583. In our agent the reranker runs as the top STRONGER step after the filter is applied the reranker only adds up cohere call and latency for no measured gain.

**What I would ship.**

The diagnosis agent for the cost of one question (10/14 vs 11/14) nearly halves latency and chat calls (~46% and ~45% respectively) which might save some tokens when ran on large scale and get faster response so we choose it. Fewer chat calls matters as the Gemini free tier allows 500 a day, and the ladder’s 4.7 per question uses that up almost twice as fast. Before shipping I would make one change, a confidence floor on sufficient like the one absent already has, since the question diagnosis lost was a 0.35 sufficient that stopped after one search. And I would want to know two things about my users’ questions first. One is how many are comparative, because routing and the plan/execute path matter most there and I only evaluated six. The other is how many ask for a reported figure, since those skip retrieval and go straight to the financials file, which the agent evaluation didn’t test.
