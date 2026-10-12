"""Bundle and skops-schema rewrites for sklearn validation features."""

import io
import json
from zipfile import ZipFile

from mlbundle import MemberName
from mlbundle.container import read_container
from mlbundle.writing import write_container


def rewrite(path, payload):
    manifest, members = read_container(path)
    del members[MemberName.MANIFEST]
    members[MemberName.SKOPS] = payload
    return write_container(manifest, members)


def unpack(data):
    with ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def pack(entries):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return buffer.getvalue()


def alter_schema(data, change):
    entries = unpack(data)
    schema = json.loads(entries["schema.json"])
    change(schema, entries)
    entries["schema.json"] = json.dumps(schema).encode()
    return pack(entries)
