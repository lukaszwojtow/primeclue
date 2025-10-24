#!/bin/sh
set -e

export NODE_OPTIONS=--openssl-legacy-provider
npm install
npm run build
