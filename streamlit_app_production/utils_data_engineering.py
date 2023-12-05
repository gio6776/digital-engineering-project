def get_table_schema(client, table_id):
    if "`" in table_id:
        table_id = table_id.replace("`", "")
    table_ref = client.get_table(table_id)
    schema_str = ""
    schema_info = {}
    for field in table_ref.schema:
        schema_str += f"{field.name} ({field.field_type}), "
        schema_info[field.name] = field.field_type

    return schema_str[:-1], schema_info