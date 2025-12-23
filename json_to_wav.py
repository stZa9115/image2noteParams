import json
import numpy as np
import soundfile as sf
import argparse
import os


def load_note_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def synthesize_note(note_data):
    fs = note_data["fs"]
    length = note_data["length"]

    t = np.arange(length) / fs
    output = np.zeros(length, dtype=np.float32)

    for p in note_data["partials"]:
        freq = p["freq"]
        alpha = np.array(p["alpha"], dtype=np.float32)
        mag = np.array(p["mag"], dtype=np.float32)

        assert alpha.shape[0] == length
        assert mag.shape[0] == length

        tone = np.sin(2 * np.pi * freq * t + alpha) * mag
        output += tone

    # normalize (optional but safe)
    # peak = np.max(np.abs(output))
    # if peak > 0:
    #     output /= peak * 1.05

    return output, fs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="note parameter json file")
    parser.add_argument("-o", "--out", help="output wav path")
    args = parser.parse_args()

    note_data = load_note_json(args.json)
    audio, fs = synthesize_note(note_data)

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.splitext(args.json)[0] + "_recon.wav"

    sf.write(out_path, audio, fs)
    print(f"wav written: {out_path}")


if __name__ == "__main__":
    main()
