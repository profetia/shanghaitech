#ifndef VM_PAGE_H
#define VM_PAGE_H

#include <hash.h>
#include <stdbool.h>
#include <stdint.h>
#include "filesys/file.h"
#include "threads/synch.h"
#include "vm/frame.h"
#include "vm/swap.h"

#define STACK_BOTTOM ((void*) 0x08048000)

enum page_location {
  PAGE_LOC_ZERO,       // Page is all zeros, no other fields are valid.
  PAGE_LOC_SWAP,       // Page is in swap, swap_index is valid.
  PAGE_LOC_MEMORY,     // Page is in memory, frame_entry is valid.
  PAGE_LOC_EXEC,       // Page is in the executable, file and file_offset are 
  // valid. Executable pages cannot be written back to the file system
  // instead they must be evicted to swap.
  PAGE_LOC_FILESYS, // Page is in the file system, file and file_offset are valid.
  PAGE_LOC_MMAPPED, // Page is in a memory mapped file, frame_entry and file are valid.
  PAGE_LOC_ERROR
};

struct sup_page_table_entry {
  uint32_t* user_vaddr;

  enum page_location location;

  struct frame_table_entry* frame_entry;
  size_t swap_index;
  struct file* file;
  off_t file_offset;
  size_t read_bytes;
  size_t zero_bytes;

  bool writable;

  bool dirty;
  bool accessed;

  struct lock* lock;
  struct hash_elem elem;
};

void sup_page_table_init(struct hash* sup_page_table);
void sup_page_table_destroy(struct hash* sup_page_table);

struct sup_page_table_entry* page_create(
    struct hash* sup_page_table, const void* user_vaddr, 
    enum page_location location, struct frame_table_entry* frame_entry, 
    size_t swap_index, struct file* file, off_t file_offset, 
    size_t read_bytes, size_t zero_bytes, bool writable);
void page_destroy(struct hash* sup_page_table, 
    struct sup_page_table_entry* entry);

struct sup_page_table_entry* page_alloc(
    struct hash* sup_page_table, const void* user_vaddr, bool writable);

struct sup_page_table_entry *page_mmap (struct hash *sup_page_table,
                                        struct file *file, const uint32_t *user_vaddr,
                                        off_t offset, size_t read_bytes,
                                        size_t zero_bytes, bool writable);

struct sup_page_table_entry* page_find(
    struct hash* sup_page_table, const void* user_vaddr);

bool page_overlaps(struct hash* sup_page_table, 
    const void* user_vaddr, size_t size);

struct sup_page_table_entry* page_pull (struct hash* sup_page_table, 
    const void* esp, const void* user_addr, bool write);

void page_unmap(struct sup_page_table_entry* spte);

void page_evict(struct sup_page_table_entry* spte);

#endif