#!/bin/sh

cat training-data-chunk-00{00..15} > training-data.ndjson
cat training-data-chunk-00{16..19} > test-data.ndjson