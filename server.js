const express = require('express');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
const { YoutubeTranscript } = require('youtube-transcript');
const { v4: uuidv4 } = require('uuid');
const ffmpeg = require('fluent-ffmpeg');

const app = express();
const PORT = 3000;
const DATA_DIR = path.join(__dirname, 'data');
const TMP_DIR  = path.join(__dirname, 'tmp');

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(TMP_DIR))  fs.mkdirSync(TMP_DIR,  { recursive: true });

/* ── Whisper pipeline (lazy-loaded, cached after first use) ── */
let _whisperPipe = null;
async function getWhisperPipe() {
  if (_whisperPipe) return _whisperPipe;
  const { pipeline, env } = await import('@xenova/transformers');
  env.cacheDir = path.join(__dirname, '.models');
  _whisperPipe = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base.en');
  return _whisperPipe;
}

/* ── Read 16-bit mono WAV → Float32Array ─────────────────── */
function readWavFloat32(filePath) {
  const buf = fs.readFileSync(filePath);
  // Walk chunks to find 'data'
  let pos = 12;
  while (pos < buf.length - 8) {
    const id = buf.slice(pos, pos + 4).toString('ascii');
    const size = buf.readUInt32LE(pos + 4);
    if (id === 'data') { pos += 8; break; }
    pos += 8 + size;
  }
  const count = Math.floor((buf.length - pos) / 2);
  const f32 = new Float32Array(count);
  for (let i = 0; i < count; i++) f32[i] = buf.readInt16LE(pos + i * 2) / 32768;
  return f32;
}

/* ── Get direct audio stream URL via yt-dlp ─────────────── */
function getStreamUrl(videoId) {
  return new Promise((resolve, reject) => {
    execFile(
      'yt-dlp',
      ['-g', '-f', 'bestaudio', '--no-playlist', '--quiet',
       `https://www.youtube.com/watch?v=${videoId}`],
      { timeout: 30000 },
      (err, stdout, stderr) => {
        const url = stdout.trim();
        // stdout에 URL이 있으면 성공 (stderr 경고는 무시)
        if (url) return resolve(url);
        reject(new Error(stderr?.trim().slice(0, 200) || err?.message || '스트림 URL을 가져오지 못했습니다.'));
      }
    );
  });
}

/* ── Download audio segment as 16 kHz mono WAV ──────────── */
async function downloadAudioWav(videoId, startSec, endSec) {
  const outWav = path.join(TMP_DIR, `${videoId}_${Date.now()}.wav`);
  const streamUrl = await getStreamUrl(videoId);

  return new Promise((resolve, reject) => {
    let cmd = ffmpeg(streamUrl);

    // seekInput on an HTTP URL uses range requests → efficient + accurate
    if (startSec > 0) cmd = cmd.seekInput(startSec);
    if (endSec != null) cmd = cmd.duration(endSec - (startSec || 0));

    cmd.audioFrequency(16000)
       .audioChannels(1)
       .audioCodec('pcm_s16le')
       .format('wav')
       .save(outWav)
       .on('end',   () => resolve(outWav))
       .on('error', (e) => reject(new Error('ffmpeg 오류: ' + e.message)));
  });
}

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

function parseYouTubeUrl(url) {
  let videoId = null;
  let isShorts = false;
  try {
    const u = new URL((url || '').trim());
    if (u.hostname.includes('youtube.com')) {
      if (u.pathname.startsWith('/shorts/')) {
        isShorts = true;
        videoId = u.pathname.split('/shorts/')[1].split('?')[0].split('/')[0];
      } else if (u.pathname === '/watch') {
        videoId = u.searchParams.get('v');
      } else if (u.pathname.startsWith('/embed/')) {
        videoId = u.pathname.split('/embed/')[1].split('?')[0];
      }
    } else if (u.hostname === 'youtu.be') {
      videoId = u.pathname.substring(1).split('?')[0];
    }
  } catch {
    const trimmed = (url || '').trim();
    if (/^[a-zA-Z0-9_-]{11}$/.test(trimmed)) videoId = trimmed;
  }
  return { videoId: videoId || null, isShorts };
}

// Merge short transcript fragments into complete sentences.
// Splits on: sentence-ending punctuation OR gap > GAP_MS between entries.
function mergeIntoSentences(items) {
  if (!items.length) return [];
  const GAP_MS  = 800;   // pause longer than this → new sentence
  const MAX_MS  = 9000;  // hard cap so no sentence exceeds ~9 s

  const result = [];
  let cur = null;

  for (const item of items) {
    if (!cur) { cur = { ...item }; continue; }

    const gap         = item.offset - (cur.offset + cur.duration);
    const mergedLen   = (item.offset + item.duration) - cur.offset;
    const endsWithPunct = /[.!?…]\s*$/.test(cur.text);

    if (endsWithPunct || gap > GAP_MS || mergedLen > MAX_MS) {
      result.push(cur);
      cur = { ...item };
    } else {
      cur.text     = cur.text + ' ' + item.text;
      cur.duration = mergedLen;
    }
  }
  if (cur) result.push(cur);
  return result;
}

function cleanText(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&apos;/g, "'")
    .replace(/\n/g, ' ').trim();
}

// GET /api/packs — list all packs
app.get('/api/packs', (req, res) => {
  try {
    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));
    const packs = files.map(file => {
      try {
        const d = JSON.parse(fs.readFileSync(path.join(DATA_DIR, file), 'utf-8'));
        return { id: d.id, title: d.title, videoId: d.videoId, isShorts: d.isShorts,
                 sentenceCount: d.transcript.length, createdAt: d.createdAt };
      } catch { return null; }
    }).filter(Boolean);
    packs.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    res.json(packs);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// GET /api/packs/:id
app.get('/api/packs/:id', (req, res) => {
  try {
    const fp = path.join(DATA_DIR, `${req.params.id}.json`);
    if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Not found' });
    res.json(JSON.parse(fs.readFileSync(fp, 'utf-8')));
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// POST /api/packs — create new pack
app.post('/api/packs', (req, res) => {
  try {
    const { title, videoId, isShorts, startTime, endTime, transcript } = req.body;
    if (!videoId) return res.status(400).json({ error: '비디오 ID가 없습니다.' });
    if (!Array.isArray(transcript) || transcript.length === 0)
      return res.status(400).json({ error: '문장이 없습니다.' });
    const id = uuidv4();
    const pack = {
      id, title: title || 'Untitled', videoId, isShorts: !!isShorts,
      startTime: startTime != null && startTime !== '' ? Number(startTime) : null,
      endTime: endTime != null && endTime !== '' ? Number(endTime) : null,
      transcript, createdAt: new Date().toISOString()
    };
    fs.writeFileSync(path.join(DATA_DIR, `${id}.json`), JSON.stringify(pack, null, 2));
    res.json(pack);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// PUT /api/packs/:id — update pack (title, transcript, etc.)
app.put('/api/packs/:id', (req, res) => {
  try {
    const fp = path.join(DATA_DIR, `${req.params.id}.json`);
    if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Not found' });
    const existing = JSON.parse(fs.readFileSync(fp, 'utf-8'));
    const updated  = { ...existing, ...req.body, id: existing.id, createdAt: existing.createdAt };
    fs.writeFileSync(fp, JSON.stringify(updated, null, 2));
    res.json(updated);
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// DELETE /api/packs/:id
app.delete('/api/packs/:id', (req, res) => {
  try {
    const fp = path.join(DATA_DIR, `${req.params.id}.json`);
    if (!fs.existsSync(fp)) return res.status(404).json({ error: 'Not found' });
    fs.unlinkSync(fp);
    res.json({ success: true });
  } catch (err) { res.status(500).json({ error: err.message }); }
});

// POST /api/transcript — fetch YouTube transcript
app.post('/api/transcript', async (req, res) => {
  const { url, startTime, endTime } = req.body;
  const { videoId, isShorts } = parseYouTubeUrl(url || '');
  if (!videoId) return res.status(400).json({ error: '유효한 YouTube URL이 아닙니다.' });

  try {
    const raw = await YoutubeTranscript.fetchTranscript(videoId);
    let items = raw
      .map(item => ({
        offset: Math.round(Number(item.offset)),
        duration: Math.round(Number(item.duration)),
        text: cleanText(item.text)
      }))
      .filter(item => item.text && !/^\[.*\]$/.test(item.text) && item.text.length > 0);

    if (startTime !== undefined && startTime !== null && startTime !== '') {
      const startMs = parseFloat(startTime) * 1000;
      items = items.filter(item => item.offset >= startMs);
    }
    if (endTime !== undefined && endTime !== null && endTime !== '') {
      const endMs = parseFloat(endTime) * 1000;
      items = items.filter(item => item.offset < endMs);
    }

    items = mergeIntoSentences(items);

    if (items.length === 0)
      return res.status(404).json({ error: '해당 구간에 자막이 없습니다.' });

    res.json({ videoId, isShorts, transcript: items });
  } catch (err) {
    res.status(500).json({
      error: '자막을 가져올 수 없습니다. 자막이 없거나 비공개 영상일 수 있습니다.'
    });
  }
});

// POST /api/parse-url — detect video type
app.post('/api/parse-url', (req, res) => {
  res.json(parseYouTubeUrl(req.body.url || ''));
});

// POST /api/align — download audio + run Whisper for accurate timestamps
app.post('/api/align', async (req, res) => {
  const { videoId, startTime, endTime } = req.body;
  if (!videoId) return res.status(400).json({ error: '비디오 ID가 없습니다.' });

  const startSec = startTime != null && startTime !== '' ? Number(startTime) : 0;
  const endSec   = endTime   != null && endTime   !== '' ? Number(endTime)   : null;

  let audioPath = null;
  try {
    /* 1. Download */
    audioPath = await downloadAudioWav(videoId, startSec, endSec);

    /* 2. WAV → Float32 (Node.js에는 AudioContext가 없어 직접 파싱) */
    const audioData = readWavFloat32(audioPath);

    /* 3. Whisper (첫 실행 시 모델 다운로드 ~142 MB) */
    const pipe   = await getWhisperPipe();
    const result = await pipe(
      { array: audioData, sampling_rate: 16000 },
      { return_timestamps: true, language: 'english', chunk_length_s: 30, stride_length_s: 5 }
    );

    /* 3. Convert chunks → transcript (timestamps are seconds from audio start) */
    const baseMs = startSec * 1000;
    const transcript = (result.chunks || [])
      .filter(c => c.text && c.text.trim() && c.timestamp?.[1] != null)
      .map(c => ({
        offset:   Math.round(c.timestamp[0] * 1000) + baseMs,
        duration: Math.round((c.timestamp[1] - c.timestamp[0]) * 1000),
        text:     c.text.trim().replace(/^\[.*?\]$/, '').trim(),
      }))
      .filter(c => c.text.length > 1 && c.duration > 0);

    if (!transcript.length) throw new Error('음성을 인식하지 못했습니다.');
    res.json({ transcript });

  } catch (err) {
    res.status(500).json({ error: err.message });
  } finally {
    if (audioPath) try { fs.unlinkSync(audioPath); } catch {}
  }
});

app.listen(PORT, () => {
  console.log(`\n🎧  YouTube 영어 받아쓰기`);
  console.log(`📡  http://localhost:${PORT}\n`);
});
