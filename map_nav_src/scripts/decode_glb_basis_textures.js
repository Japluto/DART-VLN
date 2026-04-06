#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function usage() {
  console.error("usage: decode_glb_basis_textures.js <glb_path> <out_dir> <deps_dir>");
  process.exit(2);
}

if (process.argv.length < 5) {
  usage();
}

const glbPath = process.argv[2];
const outDir = process.argv[3];
const depsDir = process.argv[4];

const basisModulePath = path.join(
  depsDir,
  "node_modules",
  "basis_universal_wasm",
  "dist",
  "transcoder",
  "basis_transcoder.js"
);
const pngJsPath = path.join(depsDir, "node_modules", "pngjs");

if (!fs.existsSync(basisModulePath)) {
  throw new Error(`basis decoder not found: ${basisModulePath}`);
}
if (!fs.existsSync(pngJsPath)) {
  throw new Error(`pngjs not found: ${pngJsPath}`);
}

const BASIS = require(basisModulePath);
const { PNG } = require(pngJsPath);

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function parseGlb(glbBytes) {
  const magic = glbBytes.readUInt32LE(0);
  if (magic !== 0x46546c67) {
    throw new Error("invalid glb magic");
  }

  let offset = 12;
  let gltf = null;
  let binChunk = null;
  while (offset < glbBytes.length) {
    const chunkLen = glbBytes.readUInt32LE(offset);
    const chunkType = glbBytes.readUInt32LE(offset + 4);
    offset += 8;
    const chunk = glbBytes.slice(offset, offset + chunkLen);
    offset += chunkLen;
    if (chunkType === 0x4e4f534a) {
      gltf = JSON.parse(chunk.toString("utf8").replace(/\0+$/, ""));
    } else if (chunkType === 0x004e4942) {
      binChunk = chunk;
    }
  }

  if (!gltf || !binChunk) {
    throw new Error("incomplete glb");
  }
  return { gltf, binChunk };
}

function writePngFromRgba(outPath, width, height, rgbaBuffer) {
  return new Promise((resolve, reject) => {
    const png = new PNG({ width, height });
    png.data = Buffer.from(rgbaBuffer);
    png.pack().pipe(fs.createWriteStream(outPath)).on("finish", resolve).on("error", reject);
  });
}

async function main() {
  ensureDir(outDir);

  const basisModule = await BASIS();
  basisModule.initializeBasis();

  const glbBytes = fs.readFileSync(glbPath);
  const { gltf, binChunk } = parseGlb(glbBytes);
  const manifest = [];

  for (let imageIndex = 0; imageIndex < (gltf.images || []).length; imageIndex += 1) {
    const image = gltf.images[imageIndex];
    const safeName = (image.name || `image_${imageIndex}`).replace(/[^a-zA-Z0-9._-]/g, "_");
    const outPath = path.join(outDir, `${String(imageIndex).padStart(3, "0")}_${safeName}.png`);

    if (fs.existsSync(outPath)) {
      manifest.push({
        index: imageIndex,
        name: image.name || null,
        mimeType: image.mimeType || null,
        path: outPath,
      });
      continue;
    }

    if (typeof image.bufferView !== "number") {
      continue;
    }

    const bufferView = gltf.bufferViews[image.bufferView];
    const start = bufferView.byteOffset || 0;
    const end = start + bufferView.byteLength;
    const encoded = new Uint8Array(binChunk.slice(start, end));

    if (image.mimeType === "image/png" || image.mimeType === "image/jpeg") {
      fs.writeFileSync(outPath, Buffer.from(encoded));
    } else if (image.mimeType === "image/x-basis") {
      const basisFile = new basisModule.BasisFile(encoded);
      if (!basisFile.startTranscoding()) {
        basisFile.close();
        basisFile.delete();
        throw new Error(`startTranscoding failed for image ${imageIndex}`);
      }

      const width = basisFile.getImageWidth(0, 0);
      const height = basisFile.getImageHeight(0, 0);
      const rgbaFormat = 13; // BASIS_TEXTURE_FORMAT.cTFRGBA32
      const rgba = new Uint8Array(basisFile.getImageTranscodedSizeInBytes(0, 0, rgbaFormat));
      const ok = basisFile.transcodeImage(rgba, 0, 0, rgbaFormat, 0, 0);
      basisFile.close();
      basisFile.delete();
      if (!ok) {
        throw new Error(`transcodeImage failed for image ${imageIndex}`);
      }
      await writePngFromRgba(outPath, width, height, rgba);
    } else {
      continue;
    }

    manifest.push({
      index: imageIndex,
      name: image.name || null,
      mimeType: image.mimeType || null,
      path: outPath,
    });
  }

  const manifestPath = path.join(outDir, "manifest.json");
  fs.writeFileSync(manifestPath, JSON.stringify({ glb: glbPath, images: manifest }, null, 2));
  console.log(manifestPath);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
