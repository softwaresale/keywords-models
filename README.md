
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

TF-IDF Accuracy:

<img width="716" alt="IMG_4700" src="https://github.com/user-attachments/assets/dd7156c0-7e6a-49fd-a2c4-db342eb7d98d" />

T5 Accuracy:

Abstract: 0.384

Body: 0.352
