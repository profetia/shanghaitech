#include "filesys/cache.h"
#include <debug.h>
#include <list.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include "filesys/filesys.h"
#include "threads/malloc.h"
#include "threads/synch.h"
#include "threads/thread.h"

static struct cache_entry cache[CACHE_SIZE];
static struct lock cache_lock;

static struct list read_ahead_list;
static struct lock read_ahead_lock;
static struct condition read_ahead_cond;
static void cache_read_ahead_daemon(void *aux UNUSED);

static struct list write_behind_list;
static struct lock write_behind_lock;
static void cache_write_behind_daemon(void *aux UNUSED);

static struct cache_entry* cache_pull(block_sector_t sector);
static struct cache_entry* cache_find_victim(void);
static void cache_evict(struct cache_entry* victim);

void 
cache_init(void) 
{
  for (size_t i = 0; i < CACHE_SIZE; ++i) 
    {
      cache[i].sector = -1;
      cache[i].dirty = false;
      cache[i].valid = false;
      cache[i].accessed = false;
      lock_init(&cache[i].lock);
    }
  lock_init(&cache_lock);

  list_init(&read_ahead_list);
  lock_init(&read_ahead_lock);
  cond_init(&read_ahead_cond);
  thread_create("cache_read_ahead_daemon", PRI_DEFAULT, 
      cache_read_ahead_daemon, NULL, NOT_A_FD);

  list_init(&write_behind_list);
  lock_init(&write_behind_lock);
  thread_create("cache_write_behind_daemon", PRI_DEFAULT,
      cache_write_behind_daemon, NULL, NOT_A_FD);
}

void 
cache_read(block_sector_t sector, void *buffer) 
{
  struct cache_entry* entry = cache_pull(sector);
  lock_acquire(&entry->lock);
  memcpy(buffer, entry->data, BLOCK_SECTOR_SIZE);
  entry->accessed = true;
  lock_release(&entry->lock);
}

void
cache_write(block_sector_t sector, const void *buffer) 
{
  struct cache_entry* entry = cache_pull(sector);
  lock_acquire(&entry->lock);
  memcpy(entry->data, buffer, BLOCK_SECTOR_SIZE);
  entry->accessed = true;
  entry->dirty = true;
  lock_release(&entry->lock);
}

void
cache_flush(void) 
{
  lock_acquire(&cache_lock);
  for (size_t i = 0; i < CACHE_SIZE; ++i) 
    {
      if (cache[i].valid && cache[i].dirty) 
        {
          block_write(fs_device, cache[i].sector, cache[i].data);
          cache[i].dirty = false;
        }
    }
  lock_release(&cache_lock);
}

static struct cache_entry*
cache_pull(block_sector_t sector) 
{
  ASSERT (sector != BLOCK_SECTOR_ERROR);

  lock_acquire(&cache_lock);
  for (size_t i = 0; i < CACHE_SIZE; ++i) 
    {
      if (cache[i].valid && cache[i].sector == sector)
      {
        lock_release(&cache_lock);
        return &cache[i];
      }    
    }
  
  struct cache_entry* victim = cache_find_victim();
  cache_evict(victim);
  block_read(fs_device, sector, victim->data);
  victim->sector = sector;
  victim->valid = true;
  lock_release(&cache_lock);
  return victim;
}

static struct cache_entry*
cache_find_victim(void) 
{
  ASSERT (lock_held_by_current_thread(&cache_lock));

  // Find an invalid entry
  for (size_t i = 0; i < CACHE_SIZE; ++i)
    {
      if (!cache[i].valid) 
        return &cache[i];
    }    

  // Clock algorithm
  for (size_t i = 0; i < CACHE_SIZE; ++i)
    {
      if (cache->accessed)
        cache->accessed = false;
      else
        return &cache[i];
    }

  // Fallback to the first entry
  return &cache[0];
}

static void
cache_evict(struct cache_entry* victim) 
{
  ASSERT (victim != NULL);
  ASSERT (lock_held_by_current_thread(&cache_lock));

  if (!victim->valid)
    return;

  if (victim->dirty)
    block_write(fs_device, victim->sector, victim->data);
  victim->valid = false;
  victim->dirty = false;
  victim->accessed = false;
  victim->sector = BLOCK_SECTOR_ERROR;
}

struct read_ahead_elem
  {
    struct list_elem elem;
    block_sector_t sector; 
    // The sector to read ahead, or BLOCK_SECTOR_ERROR to terminate the daemon
  };

void
cache_read_ahead(block_sector_t sector) 
{
  struct read_ahead_elem* elem = malloc(sizeof(struct read_ahead_elem));
  if (elem == NULL) return;

  elem->sector = sector;

  lock_acquire(&read_ahead_lock);
  list_push_back(&read_ahead_list, &elem->elem);
  cond_signal(&read_ahead_cond, &read_ahead_lock);

  lock_release(&read_ahead_lock);
}

static void
cache_read_ahead_daemon(void *aux UNUSED) 
{
  lock_acquire(&read_ahead_lock);
  while (true) 
    {
      while (list_empty(&read_ahead_list))
        cond_wait(&read_ahead_cond, &read_ahead_lock);
      struct read_ahead_elem* elem = list_entry(list_pop_front(&read_ahead_list), struct read_ahead_elem, elem);
      block_sector_t sector = elem->sector;
      free(elem);
      if (sector == BLOCK_SECTOR_ERROR)
        break;
      cache_pull(sector);           
    }

  lock_release(&read_ahead_lock);
}

struct write_behind_elem
  {
    struct list_elem elem;
    bool terminate;    
  };

void
cache_write_behind(bool terminate) 
{
  struct write_behind_elem* elem = malloc(sizeof(struct write_behind_elem));
  if (elem == NULL) return;

  elem->terminate = terminate;

  lock_acquire(&write_behind_lock);
  list_push_back(&write_behind_list, &elem->elem);
  lock_release(&write_behind_lock);
}

static void
cache_write_behind_daemon(void *aux UNUSED) 
{
  while (true)
    {
      lock_acquire(&write_behind_lock);
      if (!list_empty(&write_behind_list))
        {
          struct write_behind_elem* elem = list_entry(list_pop_front(&write_behind_list), struct write_behind_elem, elem);
          bool terminate = elem->terminate;
          free(elem);
          if (terminate)
            break;
        }
      lock_release(&write_behind_lock);

      cache_flush();
      thread_sleep(CACHE_FLUSH_INTERVAL);
    }
  
  lock_release(&write_behind_lock);
}
