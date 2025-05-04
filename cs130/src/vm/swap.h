#ifndef VM_SWAP_H
#define VM_SWAP_H

#include <bitmap.h>

#define PAGE_BLOCK_SIZE (PGSIZE / BLOCK_SECTOR_SIZE)

void swap_init(void);

size_t swap_evict(uint8_t* frame);
void swap_reclaim(uint8_t* frame, size_t index);
void swap_free(size_t index);

#endif // VM_SWAP_H