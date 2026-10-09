FROM python:3.11-slim

# The opponent is part of the environment under test, so its version has to be
# held fixed across runs that will be compared. Debian's `stockfish` is a moving
# target, so either pin it here (`--build-arg STOCKFISH_VERSION=16.1-1`, find the
# value with `apt-cache policy stockfish` in the built image) or build the image
# once and reuse that image everywhere.
ARG STOCKFISH_VERSION=17-1
RUN apt-get update && apt-get install -y \
    stockfish${STOCKFISH_VERSION:+=$STOCKFISH_VERSION} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /env/game
COPY game.py /env/game.py
RUN chmod +x /env/game.py && chown -R games:games /env

WORKDIR /env
ENV PYTHONUNBUFFERED=1
ENV PATH=$PATH:/usr/games
