#!/usr/bin/env python3
"""
cb2nrrd.py - make Shimadzu inspeXio micro-CT reconstructions (*.cb slice stacks) loadable in 3D Slicer.

Each NNNN.cb file is one reconstructed slice stored as an uncompressed little-endian TIFF.
The voxel size is read from NNNN.txt ("PixelSize(mm/pixel)=...").

Two output modes:
  header (default) - writes a tiny detached NRRD header (.nhdr) that points at the .cb files.
                     No pixel data is copied; load the .nhdr in Slicer (drag-and-drop or Add Data).
  full             - writes one self-contained .nrrd (optionally gzip-compressed) for sharing/archiving.

Usage examples
  python cb2nrrd.py  D:\\scans\\lot261019_M1                 # every reconstruction folder found below this path
  python cb2nrrd.py  D:\\scans\\lot261019_M1 --mode full --gzip
  python cb2nrrd.py  some_folder --slice-spacing 0.0444      # override if slices are not isotropic
  python cb2nrrd.py  some_folder --dry-run                   # just report what was found

Runs with plain Python 3 (no packages needed for header mode; numpy only for --mode full)
or inside 3D Slicer's Python console:
  import sys; sys.argv=['cb2nrrd.py', r'D:\\scans\\lot261019_M1']; exec(open(r'C:\\path\\to\\cb2nrrd.py').read())
"""
import argparse, os, re, struct, sys, gzip

TIFF_TYPES = {1: ('B', 1), 3: ('H', 2), 4: ('I', 4), 16: ('Q', 8)}
NRRD_TYPE = {(8, 1): 'uint8', (8, 2): 'int8', (16, 1): 'uint16', (16, 2): 'int16',
             (32, 1): 'uint32', (32, 2): 'int32', (32, 3): 'float'}


def read_tiff_layout(path):
    """Return width, height, bits, sample format, compression, strip offsets/counts, endian of a 1-page TIFF."""
    with open(path, 'rb') as f:
        head = f.read(8)
        if head[:2] == b'II': e = '<'
        elif head[:2] == b'MM': e = '>'
        else: raise ValueError(f'{path}: not a TIFF file')
        magic, ifd = struct.unpack(e + 'HI', head[2:8])
        if magic != 42: raise ValueError(f'{path}: unsupported TIFF variant (BigTIFF?)')
        f.seek(ifd)
        (n,) = struct.unpack(e + 'H', f.read(2))
        tags = {}
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(e + 'HHI4s', f.read(12))
            if typ not in TIFF_TYPES: continue
            code, size = TIFF_TYPES[typ]
            if cnt * size <= 4:
                vals = struct.unpack(e + code * cnt, val[:cnt * size])
            else:
                (off,) = struct.unpack(e + 'I', val)
                here = f.tell(); f.seek(off)
                vals = struct.unpack(e + code * cnt, f.read(cnt * size)); f.seek(here)
            tags[tag] = vals
    g = lambda t, d=None: tags.get(t, (d,))
    return dict(width=g(256)[0], height=g(257)[0], bits=g(258, 8)[0], sfmt=g(339, 1)[0],
                compression=g(259, 1)[0], spp=g(277, 1)[0],
                offsets=tags.get(273, ()), counts=tags.get(279, ()), endian=e)


def read_pixel_size(folder, first_stem):
    """mm/pixel from NNNN.txt; returns None if not found."""
    for name in [first_stem + '.txt'] + sorted(x for x in os.listdir(folder) if x.lower().endswith('.txt')):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            m = re.search(r'PixelSize\(mm/pixel\)\s*=\s*([0-9.eE+-]+)', open(p, errors='ignore').read())
            if m: return float(m.group(1))
    return None


def find_stacks(root):
    """Yield folders that contain numbered .cb slices."""
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        if any(re.fullmatch(r'\d+\.cb', f, re.I) for f in files):
            yield dirpath


def analyse(folder):
    slices = sorted((f for f in os.listdir(folder) if re.fullmatch(r'\d+\.cb', f, re.I)),
                    key=lambda f: int(f.split('.')[0]))
    nums = [int(f.split('.')[0]) for f in slices]
    gaps = [b for a, b in zip(nums, nums[1:]) if b != a + 1]
    L = read_tiff_layout(os.path.join(folder, slices[0]))
    if L['spp'] != 1: raise ValueError('multi-channel slices are not supported')
    nbytes = L['width'] * L['height'] * L['bits'] // 8
    contiguous = (L['compression'] == 1 and L['offsets'] and
                  all(L['offsets'][i] + L['counts'][i] == L['offsets'][i + 1] for i in range(len(L['offsets']) - 1)) and
                  sum(L['counts']) == nbytes)
    # every slice must share size and layout for a detached header to be valid
    size0 = os.path.getsize(os.path.join(folder, slices[0]))
    uniform = all(os.path.getsize(os.path.join(folder, s)) == size0 for s in slices)
    if uniform:
        for s in slices[:: max(1, len(slices) // 10)] + [slices[-1]]:
            Ls = read_tiff_layout(os.path.join(folder, s))
            if (Ls['offsets'][:1], Ls['width'], Ls['height'], Ls['bits']) != (L['offsets'][:1], L['width'], L['height'], L['bits']):
                uniform = False; break
    return dict(folder=folder, slices=slices, gaps=gaps, layout=L, nbytes=nbytes,
                contiguous=contiguous, uniform=uniform, offset=L['offsets'][0] if L['offsets'] else None,
                pixel=read_pixel_size(folder, slices[0].split('.')[0]))


def nrrd_header(info, sp_xy, sp_z, data_lines):
    L = info['layout']
    ntype = NRRD_TYPE.get((L['bits'], L['sfmt']))
    if ntype is None: raise ValueError(f"unsupported pixel type: {L['bits']} bit, format {L['sfmt']}")
    h = ['NRRD0004',
         f"# Shimadzu inspeXio .cb reconstruction: {info['folder']}",
         f'type: {ntype}', 'dimension: 3', 'space: left-posterior-superior',
         f"sizes: {L['width']} {L['height']} {len(info['slices'])}",
         f'space directions: ({sp_xy},0,0) (0,{sp_xy},0) (0,0,{sp_z})',
         'space origin: (0,0,0)', 'space units: "mm" "mm" "mm"', 'kinds: domain domain domain']
    if L['bits'] > 8: h.append('endian: ' + ('little' if L['endian'] == '<' else 'big'))
    return '\n'.join(h + data_lines) + '\n'


def output_name(folder):
    """Readable name: <scan folder name>[_<subfolder>] (e.g. lot..._cb_000_noring)."""
    parent, leaf = os.path.split(os.path.abspath(folder))
    pleaf = os.path.basename(parent)
    base = leaf if not pleaf or leaf == pleaf else f'{pleaf}_{leaf.split("_")[0]}'
    return re.sub(r'[^\w.+-]', '_', base)


def convert(info, mode, sp_z_override, gz, force):
    L, folder = info['layout'], info['folder']
    sp_xy = info['pixel']
    if sp_xy is None:
        raise ValueError('pixel size not found (no NNNN.txt with PixelSize) - pass --pixel-size')
    sp_z = sp_z_override or sp_xy
    name = output_name(folder)
    if mode == 'header':
        if not (info['contiguous'] and info['uniform']):
            raise ValueError('slices are compressed or differ in layout - use --mode full')
        out = os.path.join(folder, name + '.nhdr')
        lines = ['encoding: raw', f"byte skip: {info['offset']}", 'data file: LIST'] + info['slices']
    else:
        out = os.path.join(folder, name + '.nrrd')
        lines = ['encoding: ' + ('gzip' if gz else 'raw')]
    if os.path.exists(out) and not force:
        print(f'  exists, skipped (use --force): {out}'); return out
    hdr = nrrd_header(info, sp_xy, sp_z, lines)
    if mode == 'header':
        with open(out, 'w', newline='\n') as f: f.write(hdr)
    else:
        import numpy as np  # only needed here
        dt = np.dtype(L['endian'] + {8: 'u1', 16: 'u2', 32: 'u4'}[L['bits']] if L['sfmt'] != 3 else L['endian'] + 'f4')
        with open(out, 'wb') as f:
            f.write(hdr.encode() + b'\n')
            sink = gzip.GzipFile(fileobj=f, mode='wb', compresslevel=1) if gz else f
            for i, s in enumerate(info['slices']):
                p = os.path.join(folder, s)
                Ls = read_tiff_layout(p)
                if Ls['compression'] != 1: raise ValueError(f'{s}: compressed TIFF not supported in this script')
                buf = bytearray()
                with open(p, 'rb') as src:
                    for o, c in zip(Ls['offsets'], Ls['counts']):
                        src.seek(o); buf += src.read(c)
                a = np.frombuffer(bytes(buf[:info['nbytes']]), dtype=dt)
                sink.write(a.tobytes())
                if i % 100 == 0: print(f'    slice {i}/{len(info["slices"])}', flush=True)
            if gz: sink.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='a scan folder, or any folder above several scans')
    ap.add_argument('--mode', choices=['header', 'full'], default='header')
    ap.add_argument('--gzip', action='store_true', help='compress the full .nrrd (smaller, slower to load)')
    ap.add_argument('--pixel-size', type=float, help='mm/pixel in-plane, if no NNNN.txt is present')
    ap.add_argument('--slice-spacing', type=float, help='mm between slices (default: same as pixel size)')
    ap.add_argument('--force', action='store_true', help='overwrite existing outputs')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    stacks = list(find_stacks(a.path))
    if not stacks: sys.exit(f'No numbered .cb slices found under {a.path}')
    for folder in stacks:
        print(f'\n{folder}')
        try:
            info = analyse(folder)
            if a.pixel_size: info['pixel'] = a.pixel_size
            L = info['layout']
            print(f"  {len(info['slices'])} slices, {L['width']}x{L['height']}, {L['bits']}-bit, "
                  f"voxel {info['pixel']} mm (z {a.slice_spacing or info['pixel']} mm), "
                  f"header {info['offset']} B, contiguous={info['contiguous']}, uniform={info['uniform']}")
            if info['gaps']: print(f"  WARNING: slice numbering has gaps (first at {info['gaps'][0]})")
            if not a.dry_run:
                print('  wrote', convert(info, a.mode, a.slice_spacing, a.gzip, a.force))
        except Exception as ex:
            print('  SKIPPED:', ex)


if __name__ == '__main__':
    main()
