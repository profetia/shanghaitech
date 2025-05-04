ALTER TABLE "Horras" ADD starting_lng_group FLOAT;
ALTER TABLE "Horras" ADD starting_lat_group FLOAT;
ALTER TABLE "Horras" ADD dest_lng_group FLOAT;
ALTER TABLE "Horras" ADD dest_lat_group FLOAT;

UPDATE "Horras" SET dest_lat_group = CAST(dest_lat AS DECIMAL(10, 2)), dest_lng_group = CAST(dest_lng AS DECIMAL(10, 2)), starting_lat_group = CAST(starting_lat AS DECIMAL(10, 2)), starting_lng_group = CAST(starting_lng AS DECIMAL(10, 2));