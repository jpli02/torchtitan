# Sandboxed IFBench verifier image: the official allenai/IFBench checkout (build context) with its dependencies and the
# NLTK data it downloads at first use, so grading runs with --network none.
#   docker build -t ifbench-grader -f boptim_lcb/ifbench_grader.Dockerfile /home/jli199/boptim_scratch/ifbench_dl/IFBench-main
#   docker run --rm --network none -v /srv/scratch/jli199/ifb_grade_stage:/w ifbench-grader python /w/ifb_grade.py /w/ifbench.jsonl /w/GEN.jsonl /w/OUT.json
FROM python:3.12-slim
RUN pip install --no-cache-dir absl-py langdetect nltk immutabledict emoji "syllapy>=0.8.0" spacy
COPY . /app
WORKDIR /app
ENV NLTK_DATA=/app/ifbench/.nltk_data
RUN python -c "import nltk; [nltk.download(p, download_dir='/app/ifbench/.nltk_data', quiet=True) for p in ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng']]" \
    && python -c "import sys; sys.path.insert(0, '/app'); import evaluation_lib; from ifbench import instructions_registry; print(len(instructions_registry.INSTRUCTION_DICT), 'verifiers')"
