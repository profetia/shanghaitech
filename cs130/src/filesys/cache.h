#ifndef FILESYS_CACHE_H
#define FILESYS_CACHE_H

#include <stdbool.h>
#include <stdint.h>
#include "devices/block.h"
#include "threads/synch.h"

#define BLOCK_SECTOR_ERROR 0xffffffff

#define CACHE_SIZE 64
#define CACHE_FLUSH_INTERVAL 10000

struct cache_entry 
  {
    block_sector_t sector;
    bool dirty;
    bool valid;
    bool accessed;
    uint8_t data[BLOCK_SECTOR_SIZE];
    struct lock lock;
  };

void cache_init(void);
void cache_read(block_sector_t sector, void *buffer);
void cache_write(block_sector_t sector, const void *buffer);
void cache_flush(void);

void cache_read_ahead(block_sector_t sector);
void cache_write_behind(bool terminate);


#endif // FILESYS_CACHE_H