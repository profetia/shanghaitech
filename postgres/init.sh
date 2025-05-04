#!/bin/bash
set -e

pg_restore -d $POSTGRES_DB /docker-entrypoint-initdb.d/db.dump
