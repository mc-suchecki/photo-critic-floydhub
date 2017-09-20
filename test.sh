#!/bin/bash

rm output.json && curl -o output.json -F "file=@./test.jpg" https://www.floydhub.com/expose/xDR5RJeQB3Y33bdLfo9xrX && cat output.json

