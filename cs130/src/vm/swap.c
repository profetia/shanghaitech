#include "vm/swap.h"
#include "devices/block.h"
#include "threads/synch.h"
#include "threads/vaddr.h"

static struct block* swap_block;
static struct bitmap* swap_bitmap;
static struct lock swap_lock;

static size_t swap_size (void);
static void read_from_block (const uint8_t *frame, size_t index);
static void write_to_block (uint8_t* frame, size_t index);

void swap_init(void) 
{
  swap_block = block_get_role(BLOCK_SWAP);
  swap_bitmap = bitmap_create(swap_size());
  lock_init(&swap_lock);
}

static size_t
swap_size(void) 
{
  return block_size(swap_block) / PAGE_BLOCK_SIZE;
}

static void
read_from_block (const uint8_t *frame, size_t index)
{
  ASSERT(frame != NULL);
  ASSERT(index < swap_size());
  ASSERT(index != BITMAP_ERROR);
  ASSERT(bitmap_test(swap_bitmap, index));

  lock_acquire(&swap_lock);
  for (size_t i = 0; i < PAGE_BLOCK_SIZE; i++) 
    {
        block_read(swap_block, PAGE_BLOCK_SIZE * index + i, 
            frame + (i * BLOCK_SECTOR_SIZE));
    }
  lock_release(&swap_lock);
}

static void
write_to_block(uint8_t* frame, size_t index)
{
  ASSERT(frame != NULL);
  ASSERT(index < swap_size());
  ASSERT(index != BITMAP_ERROR);
  ASSERT(bitmap_test(swap_bitmap, index));

  lock_acquire(&swap_lock);
  for (int i = 0; i < PAGE_BLOCK_SIZE; i++) 
    {
        block_write(swap_block, PAGE_BLOCK_SIZE * index + i, frame + (i * BLOCK_SECTOR_SIZE));
    }
  lock_release(&swap_lock);
}

size_t 
swap_evict(uint8_t* frame) 
{
  ASSERT(frame != NULL);

  lock_acquire(&swap_lock);
  size_t index = bitmap_scan_and_flip(swap_bitmap, 0, 1, false);
  ASSERT(index != BITMAP_ERROR);
  lock_release(&swap_lock);

  write_to_block(frame, index);
  return index;
}

void
swap_reclaim(uint8_t* frame, size_t index) 
{
  ASSERT(frame != NULL);
  ASSERT(index < swap_size());
  ASSERT(bitmap_test(swap_bitmap, index));

  read_from_block(frame, index);
  lock_acquire(&swap_lock);
  bitmap_reset(swap_bitmap, index);
  lock_release(&swap_lock);
}

void
swap_free(size_t index) 
{
  ASSERT(index < swap_size());
  ASSERT(bitmap_test(swap_bitmap, index));

  lock_acquire(&swap_lock);
  bitmap_reset(swap_bitmap, index);
  lock_release(&swap_lock);
}