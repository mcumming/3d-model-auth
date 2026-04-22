"""
STL File Signing and Verification Module
=========================================

Provides digital signature embedding and extraction for STL (Stereolithography)
3D model files. Supports both ASCII and binary STL formats.

Approach:
- Signature bits are embedded in the least significant bits of vertex
  z-coordinates using IEEE 754 float manipulation (same technique as OBJ).
- Metadata (artist info, original hash, method marker) is stored in:
  - Binary STL: the 80-byte header
  - ASCII STL: the solid name line
"""

import struct
import hashlib
import json
import base64
import io
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# STL parsing helpers
# ---------------------------------------------------------------------------

_BINARY_TRAIL_DELIMITER = b"\n##3DAUTH_META##\n"


def _is_binary_stl(data: bytes) -> bool:
    """Heuristic to distinguish binary STL from ASCII STL.

    Binary STL: 80-byte header + 4-byte uint32 triangle count + triangles.
    Each triangle is 50 bytes (12 normal + 36 vertex + 2 attribute).
    Signed binary STL may also have trailing metadata after the delimiter.
    """
    if len(data) < 84:
        return False

    # Strip any trailing metadata we may have appended
    core_data = data.split(_BINARY_TRAIL_DELIMITER)[0] if _BINARY_TRAIL_DELIMITER in data else data

    # Check if it starts with "solid" and looks like ASCII
    try:
        text_start = core_data[:80].decode("ascii", errors="ignore").strip().lower()
    except Exception:
        text_start = ""

    # Even binary files sometimes start with "solid" in the header,
    # so also check that the expected size matches binary format.
    num_triangles = struct.unpack_from("<I", core_data, 80)[0]
    expected_size = 84 + num_triangles * 50
    if expected_size == len(core_data):
        return True

    # If we have "solid" text followed by actual ASCII facet data, treat as ASCII
    if text_start.startswith("solid"):
        # Look for "facet" keyword after initial solid line
        try:
            first_kb = core_data[:1024].decode("ascii", errors="ignore").lower()
            if "facet" in first_kb:
                return False
        except Exception:
            pass

    return True


def _parse_binary_stl(data: bytes) -> Tuple[bytes, int, List[dict]]:
    """Parse a binary STL file.

    Returns:
        (header_80_bytes, num_triangles, list_of_triangles)
        Each triangle dict: {
            'normal': (nx, ny, nz),
            'vertices': [(x,y,z), (x,y,z), (x,y,z)],
            'attr': int  (attribute byte count)
        }
    """
    # Strip trailing metadata if present
    core = data.split(_BINARY_TRAIL_DELIMITER)[0] if _BINARY_TRAIL_DELIMITER in data else data
    header = core[:80]
    num_triangles = struct.unpack_from("<I", core, 80)[0]
    triangles: List[dict] = []
    offset = 84

    for _ in range(num_triangles):
        nx, ny, nz = struct.unpack_from("<fff", core, offset)
        offset += 12
        v1 = struct.unpack_from("<fff", core, offset)
        offset += 12
        v2 = struct.unpack_from("<fff", core, offset)
        offset += 12
        v3 = struct.unpack_from("<fff", core, offset)
        offset += 12
        attr = struct.unpack_from("<H", core, offset)[0]
        offset += 2

        triangles.append({
            "normal": (nx, ny, nz),
            "vertices": [v1, v2, v3],
            "attr": attr,
        })

    return header, num_triangles, triangles


def _write_binary_stl(header: bytes, triangles: List[dict]) -> bytes:
    """Write triangles back to binary STL bytes."""
    buf = io.BytesIO()
    buf.write(header[:80].ljust(80, b"\x00"))
    buf.write(struct.pack("<I", len(triangles)))

    for tri in triangles:
        nx, ny, nz = tri["normal"]
        buf.write(struct.pack("<fff", nx, ny, nz))
        for vx, vy, vz in tri["vertices"]:
            buf.write(struct.pack("<fff", vx, vy, vz))
        buf.write(struct.pack("<H", tri.get("attr", 0)))

    return buf.getvalue()


def _parse_ascii_stl(text: str) -> Tuple[str, List[dict]]:
    """Parse an ASCII STL file.

    Returns:
        (solid_name, list_of_triangles)
    """
    lines = text.strip().splitlines()
    solid_name = ""
    if lines and lines[0].strip().lower().startswith("solid"):
        solid_name = lines[0].strip()[5:].strip()

    triangles: List[dict] = []
    i = 1
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith("facet normal"):
            parts = line.split()
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            verts = []
            i += 1  # skip to "outer loop"
            while i < len(lines):
                vline = lines[i].strip().lower()
                if vline.startswith("vertex"):
                    vp = vline.split()
                    verts.append((float(vp[1]), float(vp[2]), float(vp[3])))
                elif vline.startswith("endloop") or vline.startswith("endfacet"):
                    pass
                if vline.startswith("endfacet"):
                    break
                i += 1
            triangles.append({
                "normal": normal,
                "vertices": verts,
                "attr": 0,
            })
        i += 1

    return solid_name, triangles


def _write_ascii_stl(solid_name: str, triangles: List[dict]) -> str:
    """Write triangles back to ASCII STL string."""
    lines = [f"solid {solid_name}"]
    for tri in triangles:
        nx, ny, nz = tri["normal"]
        lines.append(f"  facet normal {nx:.8e} {ny:.8e} {nz:.8e}")
        lines.append("    outer loop")
        for vx, vy, vz in tri["vertices"]:
            lines.append(f"      vertex {vx:.8e} {vy:.8e} {vz:.8e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {solid_name}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# IEEE 754 LSB helpers (same technique as OBJ implementation)
# ---------------------------------------------------------------------------

def _float_to_int(coord: float) -> int:
    """Convert float to IEEE 754 32-bit integer representation."""
    return struct.unpack("!I", struct.pack("!f", coord))[0]


def _int_to_float(coord_int: int) -> float:
    """Convert IEEE 754 32-bit integer back to float."""
    return struct.unpack("!f", struct.pack("!I", coord_int))[0]


def _embed_2bits(coord: float, bits: int) -> float:
    """Embed 2 bits into the LSBs of an IEEE 754 float."""
    ci = _float_to_int(coord)
    ci = (ci & ~0b11) | (bits & 0b11)
    return _int_to_float(ci)


def _extract_2bits(coord: float) -> int:
    """Extract 2 LSBs from an IEEE 754 float."""
    return _float_to_int(coord) & 0b11


# ---------------------------------------------------------------------------
# Metadata encoding helpers
# ---------------------------------------------------------------------------

_MARKER_PREFIX = "3DAuth-STL"


def _encode_metadata(artist_info: dict, original_hash: str) -> str:
    """Encode metadata to a compact string for storage in header / solid name."""
    payload = {
        "h": original_hash,
        "a": base64.b64encode(json.dumps(artist_info).encode()).decode(),
    }
    return f"{_MARKER_PREFIX}|{base64.b64encode(json.dumps(payload).encode()).decode()}"


def _decode_metadata(marker: str) -> Tuple[Optional[str], Optional[dict]]:
    """Decode metadata from header / solid name.

    Returns (original_hash, artist_info) or (None, None) on failure.
    """
    if _MARKER_PREFIX not in marker:
        return None, None
    try:
        b64_part = marker.split(f"{_MARKER_PREFIX}|", 1)[1].split("\x00")[0].strip()
        payload = json.loads(base64.b64decode(b64_part).decode())
        original_hash = payload.get("h")
        artist_info = json.loads(base64.b64decode(payload["a"]).decode())
        return original_hash, artist_info
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Public API: embed / extract for STL files
# ---------------------------------------------------------------------------

def embed_signature_stl(stl_data: bytes, signature_hex: str,
                        artist_info: dict) -> bytes:
    """Embed a digital signature into an STL file.

    Works with both ASCII and binary STL. The returned data is always in the
    same format as the input (binary in → binary out, ASCII in → ASCII out).

    Signature bits are embedded in the z-coordinates of triangle vertices
    using 2-bit LSB modification on IEEE 754 floats.

    Args:
        stl_data: Raw bytes of the STL file.
        signature_hex: Hex-encoded digital signature (e.g. RSA-2048 → 512 hex chars).
        artist_info: Dict with artist metadata (name, email, website, timestamp).

    Returns:
        Modified STL file bytes with embedded signature.

    Raises:
        ValueError: If the model does not have enough vertices.
    """
    is_binary = _is_binary_stl(stl_data)

    # --- Parse --------------------------------------------------------
    if is_binary:
        header, _num_tri, triangles = _parse_binary_stl(stl_data)
    else:
        text = stl_data.decode("utf-8", errors="replace")
        _solid_name, triangles = _parse_ascii_stl(text)

    # --- Collect all vertex z-coordinates (tri_idx, vert_idx) ---------
    all_z: List[Tuple[int, int, float]] = []
    for ti, tri in enumerate(triangles):
        for vi, (vx, vy, vz) in enumerate(tri["vertices"]):
            all_z.append((ti, vi, vz))

    # --- Compute requirements -----------------------------------------
    sig_bytes = bytes.fromhex(signature_hex)
    required_vertices = len(sig_bytes) * 4  # 2 bits per vertex, 8 bits/byte → 4 verts/byte

    if len(all_z) < required_vertices:
        raise ValueError(
            f"Model has {len(all_z)} vertices, but {required_vertices} are "
            f"needed to embed the signature"
        )

    # --- Hash original data for verification --------------------------
    original_hash = hashlib.sha256(stl_data).hexdigest()

    # --- Embed signature bits -----------------------------------------
    for byte_idx, byte_val in enumerate(sig_bytes):
        for bit_pair_idx in range(4):
            v_index = byte_idx * 4 + bit_pair_idx
            if v_index >= len(all_z):
                break

            ti, vi, z_coord = all_z[v_index]
            bit_pair = (byte_val >> (6 - bit_pair_idx * 2)) & 0b11
            z_new = _embed_2bits(z_coord, bit_pair)

            # Update triangle vertex
            vx, vy, _vz = triangles[ti]["vertices"][vi]
            triangles[ti]["vertices"][vi] = (vx, vy, z_new)

    # --- Store metadata -----------------------------------------------
    meta_str = _encode_metadata(artist_info, original_hash)

    if is_binary:
        # Keep original header (or a brief marker) and append metadata
        # as a trailing block after the standard binary data.
        stl_bytes = _write_binary_stl(header, triangles)
        return stl_bytes + _BINARY_TRAIL_DELIMITER + meta_str.encode("utf-8")
    else:
        # Use the solid name to store metadata
        return _write_ascii_stl(meta_str, triangles).encode("utf-8")


def extract_signature_stl(stl_data: bytes) -> Tuple[
    Optional[str], Optional[str], Optional[dict]
]:
    """Extract an embedded signature from an STL file.

    Returns:
        (signature_hex, original_hash, artist_info)
        All values are None if no valid signature is found.
    """
    is_binary = _is_binary_stl(stl_data)

    # --- Parse --------------------------------------------------------
    if is_binary:
        header, _num_tri, triangles = _parse_binary_stl(stl_data)
        # Metadata is stored as trailing data after the delimiter
        if _BINARY_TRAIL_DELIMITER in stl_data:
            marker = stl_data.split(_BINARY_TRAIL_DELIMITER, 1)[1].decode(
                "utf-8", errors="replace"
            )
        else:
            # Fallback: check the header
            marker = header.decode("ascii", errors="replace")
    else:
        text = stl_data.decode("utf-8", errors="replace")
        solid_name, triangles = _parse_ascii_stl(text)
        marker = solid_name

    # --- Check for our marker -----------------------------------------
    original_hash, artist_info = _decode_metadata(marker)
    if original_hash is None:
        return None, None, None

    # --- Collect vertex z-coordinates ---------------------------------
    all_z: List[float] = []
    for tri in triangles:
        for _vx, _vy, vz in tri["vertices"]:
            all_z.append(vz)

    if len(all_z) < 256 * 4:  # RSA-2048 signature = 256 bytes
        return None, None, None

    # --- Extract signature bytes (256 bytes for RSA-2048) -------------
    sig_bytes = bytearray()
    for byte_idx in range(256):
        byte_val = 0
        for bit_pair_idx in range(4):
            v_index = byte_idx * 4 + bit_pair_idx
            if v_index >= len(all_z):
                break
            bit_pair = _extract_2bits(all_z[v_index])
            byte_val |= bit_pair << (6 - bit_pair_idx * 2)
        sig_bytes.append(byte_val)

    return sig_bytes.hex(), original_hash, artist_info


def verify_stl_signature(signature_hex: str, original_hash_hex: str,
                         public_key) -> bool:
    """Verify an STL signature using the stored original file hash.

    Because the signed STL file has modified vertex z-coordinates (from LSB
    embedding), we cannot simply re-hash the signed file.  Instead we use
    the original file hash that was recorded during signing.

    Args:
        signature_hex: Hex-encoded RSA signature extracted from the STL file.
        original_hash_hex: Hex digest of the original (pre-signing) STL data,
                           as stored in the embedded metadata.
        public_key: RSA public key object.

    Returns:
        True if the signature is valid, False otherwise.
    """
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives import hashes

    try:
        signature_bytes = bytes.fromhex(signature_hex)
        original_hash_bytes = bytes.fromhex(original_hash_hex)
        public_key.verify(
            signature_bytes,
            original_hash_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False
