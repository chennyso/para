import argparse
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE/'paramorph'))

from agents import pvc_transport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mailbox',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('action',choices=('pull','push'))
    args = parser.parse_args()
    pvc_transport.MODEL = args.model
    pvc_transport.MAILBOX = args.mailbox
    if args.action=='pull':
        request = (pvc_transport.pull() if (args.mailbox/'ownership-protocol.json').is_file() else None)
        print(pvc_transport.encoded(request),flush=True)
    else:
        pvc_transport.push(json.loads(sys.stdin.read()))


if __name__=='__main__':
    main()
