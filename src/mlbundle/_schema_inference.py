"""The temporary adaptation between local mlschema inference and MLForm."""


def mlform_fields(fields: list[dict]) -> list[dict]:
    # Adapt mlschema 0.2.1 to MLForm 0.1.25 until mlschema emits series columns.
    return [
        {
            **{
                key: value
                for key, value in field.items()
                if key not in ("field1", "field2")
            },
            "columns": [
                {
                    **{
                        key: value
                        for key, value in field[axis].items()
                        if key != "mappedTo"
                    },
                    "id": axis,
                }
                for axis in ("field1", "field2")
            ],
        }
        if field.get("kind") == "series"
        else field
        for field in fields
    ]
