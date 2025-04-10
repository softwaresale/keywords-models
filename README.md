
## Data format
The training data is in NDJSON format (newline-delimited JSON). To read the file, basically read the file
line by line and parse each line. You can collect these all into a list.

Each object should be:
```json
{
  "id": "string",
  "abstract": "string",
  "keywords": ["string"],
  "body": "string"
}
```
or something like that. Check what's actually there.

