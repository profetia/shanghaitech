FROM postgres:15-alpine



COPY init.sh /docker-entrypoint-initdb.d/

COPY db.dump /docker-entrypoint-initdb.d/

COPY init.sql /docker-entrypoint-initdb.d/
