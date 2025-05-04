#include "filesys/inode.h"
#include <list.h>
#include <debug.h>
#include <round.h>
#include <stddef.h>
#include <string.h>
#include "directory.h"
#include "filesys/filesys.h"
#include "filesys/free-map.h"
#include "filesys/cache.h"
#include "stdbool.h"
#include "threads/malloc.h"


/* Identifies an inode. */
#define INODE_MAGIC 0x494e4f44
#define INDIRECT_BLOCK_MAGIC 0x494e4449
#define DOUBLE_INDIRECT_BLOCK_MAGIC 0x44424944
#define DIRECT_BLOCK_SIZE 12
#define INDIRECT_BLOCK_SIZE 127
#define DOUBLE_INDIRECT_BLOCK_SIZE 127
#define DOUBLE_INDIRECT_BLOCK_IN_INODE_SIZE 2

static char zeros[BLOCK_SECTOR_SIZE];

/* On-disk indirect block.
   Must be exactly BLOCK_SECTOR_SIZE bytes long. */
struct indirect_block_disk_t
{
  block_sector_t direct_blocks[128 - 1];
  unsigned magic;                     /* Magic number. */
};

/* On-disk double indirect block.
   Must be exactly BLOCK_SECTOR_SIZE bytes long. */
struct double_indirect_block_disk_t
{
  block_sector_t indirect_block_disk[128 - 1];
  unsigned magic;                     /* Magic number. */
};


/* On-disk inode.
   Must be exactly BLOCK_SECTOR_SIZE bytes long. */
struct inode_disk
  {
    off_t length;                       /* File size in bytes. */
    block_sector_t direct_blocks[12];  /* Direct blocks. */
    block_sector_t indirect_block;      /* Indirect block. */
    block_sector_t double_indirect_block[2]; /* Double indirect block. */
    bool isdir;                         /* Is a directory or not. default: not. */
    unsigned magic;                     /* Magic number. */
    uint32_t unused[110];               /* Not used. */
  };

static struct double_indirect_block_disk_t template_disk_double_indirect_block;
static struct indirect_block_disk_t template_disk_indirect_block;
static struct inode_disk template_disk_inode;

void template_init(void);
bool direct_block_init_if_need(block_sector_t *sector);
bool indirect_block_init_if_need(block_sector_t *sector);
bool double_indirect_block_init_if_need(block_sector_t *sector);
block_sector_t inode_seek (struct inode_disk * inode_disk, block_sector_t logical_sector);

void template_init(){
  /* init template_disk_double_indirect_block */
  for(int i = 0;i<DOUBLE_INDIRECT_BLOCK_SIZE;++i)
    template_disk_double_indirect_block.indirect_block_disk[i] = NOT_A_SECTOR;
  template_disk_double_indirect_block.magic = DOUBLE_INDIRECT_BLOCK_MAGIC;
  /* init template_disk_indirect_block */
  for(int i = 0;i<INDIRECT_BLOCK_SIZE;++i)
    template_disk_indirect_block.direct_blocks[i] = NOT_A_SECTOR;
  template_disk_indirect_block.magic = INDIRECT_BLOCK_MAGIC;
  /* init template_disk_inode */
  for(int i = 0;i<DOUBLE_INDIRECT_BLOCK_IN_INODE_SIZE;++i)
    template_disk_inode.double_indirect_block[i] = NOT_A_SECTOR;
  for(int i = 0;i<DIRECT_BLOCK_SIZE;++i)
    template_disk_inode.direct_blocks[i] = NOT_A_SECTOR;
  template_disk_inode.indirect_block = NOT_A_SECTOR;
  template_disk_inode.length = 0;
  template_disk_inode.isdir = false;
  template_disk_inode.magic = INODE_MAGIC;
}

/* Returns the number of sectors to allocate for an inode SIZE
   bytes long. */
static inline size_t
bytes_to_sectors (off_t size)
{
  return DIV_ROUND_UP (size, BLOCK_SECTOR_SIZE);
}

/* Returns the number of bytes in SECTORS sectors. */
static inline off_t
sectors_to_bytes (size_t sectors)
{
  return sectors * BLOCK_SECTOR_SIZE;
}

/* In-memory inode. */
struct inode 
  {
    struct list_elem elem;              /* Element in inode list. */
    block_sector_t sector;              /* Sector number of disk location. */
    int open_cnt;                       /* Number of openers. */
    bool removed;                       /* True if deleted, false otherwise. */
    int deny_write_cnt;                 /* 0: writes ok, >0: deny writes. */
    struct inode_disk data;             /* Inode content. */
  };


/* List of open inodes, so that opening a single inode twice
   returns the same `struct inode'. */
static struct list open_inodes;

bool direct_block_init_if_need(block_sector_t *sector){
  ASSERT(sector != NULL);
  if(*sector == NOT_A_SECTOR){
    if(!free_map_allocate(1, sector)){
      PANIC("free_map_allocate_if_need: free_map_allocate failed");
      *sector = NOT_A_SECTOR;
      return false;
    }
    cache_write(*sector, zeros);
    return true;
  }
  return false;
}

bool indirect_block_init_if_need(block_sector_t *sector){
  ASSERT(sector != NULL);
  if(*sector == NOT_A_SECTOR){
    if(!free_map_allocate(1, sector)){
      /* PANIC("indirect_block_init_if_need: free_map_allocate failed"); */
      *sector = NOT_A_SECTOR;
      return false;
    }
    cache_write(*sector, &template_disk_indirect_block);
    return true;
  }
  return false;
}

bool double_indirect_block_init_if_need(block_sector_t *sector){
  ASSERT(sector != NULL);
  if(*sector == NOT_A_SECTOR){
    if(!free_map_allocate(1, sector)){
      /* PANIC("double_indirect_block_init_if_need: free_map_allocate failed"); */
      *sector = NOT_A_SECTOR;
      return false;
    }
    cache_write(*sector, &template_disk_double_indirect_block);
    return true;
  }
  return false;
}
/*  Seek the logical_sector in the inode_disk, return the physical_sector 
    and try to create all the necessary blocks if need and write new indirect tables to the disk. 
    The root node, (ie. inode) won't be written to the disk. This should be done manually.
    This function should be called with aquiring locks of inode_disk and indirect blocks. 
    return NOT_A_SECTOR if failed.
    feature:
      - Support large sparse files. (theoretically, 2TiB)
      - Support large files. (theoretically, 2TiB)
      - Try to create all the necessary blocks if need
    caveats:
      - All indirect/double indirect blocks should be first created in this function.
      - The inode_disk should be written to the disk after this function.
    deficiency:
      - poor performance since frequent access to disks. (may be relieved by caching)
      - complex integer division and modulo calculation.
      - This function is not thread safe.
*/
block_sector_t inode_seek (struct inode_disk * inode_disk, block_sector_t logical_sector){
  ASSERT(inode_disk != NULL);
  /* seek in direct blocks */
  if(logical_sector < DIRECT_BLOCK_SIZE){
    direct_block_init_if_need(&inode_disk->direct_blocks[logical_sector]);
    return inode_disk->direct_blocks[logical_sector];
  }
  /* seek in indirect blocks */
  if(logical_sector < INDIRECT_BLOCK_SIZE + DIRECT_BLOCK_SIZE){
    /* init indirect block if need */
    indirect_block_init_if_need(&inode_disk->indirect_block);
    if(inode_disk->indirect_block == NOT_A_SECTOR)
      return NOT_A_SECTOR;
    struct indirect_block_disk_t *disk_indirect_block = NULL;
    disk_indirect_block = calloc (1, sizeof *disk_indirect_block);
    /* read indirect block from disk */
    cache_read(inode_disk->indirect_block, disk_indirect_block);
    //TODO: caching indirect block
    /* init direct block if need */
    bool write_back_disk_indirect_block = direct_block_init_if_need(&disk_indirect_block->direct_blocks[logical_sector - DIRECT_BLOCK_SIZE]);
    block_sector_t physical_sector = disk_indirect_block->direct_blocks[logical_sector - DIRECT_BLOCK_SIZE];
    if(write_back_disk_indirect_block)
      cache_write(inode_disk->indirect_block, disk_indirect_block);
    free(disk_indirect_block);
    return physical_sector;
  }
  /* seek in double indirect blocks */

  unsigned long double_indirect_block_index = (logical_sector - INDIRECT_BLOCK_SIZE - DIRECT_BLOCK_SIZE) 
                                              /
                                              (INDIRECT_BLOCK_SIZE * DOUBLE_INDIRECT_BLOCK_SIZE);
  unsigned long indirect_block_index = (logical_sector - INDIRECT_BLOCK_SIZE - DIRECT_BLOCK_SIZE) 
                                       %
                                       (INDIRECT_BLOCK_SIZE * DOUBLE_INDIRECT_BLOCK_SIZE)
                                       /
                                        INDIRECT_BLOCK_SIZE;
                                       ;
  unsigned long direct_block_index = (logical_sector - INDIRECT_BLOCK_SIZE - DIRECT_BLOCK_SIZE) 
                                     %
                                     (INDIRECT_BLOCK_SIZE * DOUBLE_INDIRECT_BLOCK_SIZE)
                                     %
                                      INDIRECT_BLOCK_SIZE;
                                      /* equal to (logical_sector - INDIRECT_BLOCK_SIZE - DIRECT_BLOCK_SIZE) % INDIRECT_BLOCK_SIZE */
  
  if(double_indirect_block_index >= DOUBLE_INDIRECT_BLOCK_IN_INODE_SIZE){
    return NOT_A_SECTOR;
  }

  /* init double indirect block if need */
  double_indirect_block_init_if_need(&inode_disk->double_indirect_block[double_indirect_block_index]);
  if(inode_disk->double_indirect_block[double_indirect_block_index] == NOT_A_SECTOR)
    return NOT_A_SECTOR;

  /* read double indirect block from disk */
  struct double_indirect_block_disk_t *disk_double_indirect_block = NULL;
  disk_double_indirect_block = calloc (1, sizeof *disk_double_indirect_block);
  cache_read(inode_disk->double_indirect_block[double_indirect_block_index], disk_double_indirect_block);
  //TODO: caching double indirect block

  /* init indirect block if need */
  bool write_back_disk_double_indirect_block = indirect_block_init_if_need(&disk_double_indirect_block->indirect_block_disk[indirect_block_index]);
  if(disk_double_indirect_block->indirect_block_disk[indirect_block_index] == NOT_A_SECTOR){
    free(disk_double_indirect_block);
    return NOT_A_SECTOR;
  }

  /* read indirect block from disk */
  struct indirect_block_disk_t *disk_indirect_block = NULL;
  disk_indirect_block = calloc (1, sizeof *disk_indirect_block);
  cache_read(disk_double_indirect_block->indirect_block_disk[indirect_block_index], disk_indirect_block);
  //TODO: caching indirect block

  /* init the direct block in the indirect block if need */
  bool write_back_disk_indirect_block = direct_block_init_if_need(&disk_indirect_block->direct_blocks[direct_block_index]);
  if(write_back_disk_indirect_block)
    cache_write(disk_double_indirect_block->indirect_block_disk[indirect_block_index], disk_indirect_block);
  if(write_back_disk_double_indirect_block)
    cache_write(inode_disk->double_indirect_block[double_indirect_block_index], disk_double_indirect_block);

  block_sector_t physical_sector = disk_indirect_block->direct_blocks[direct_block_index];

  free(disk_double_indirect_block);
  free(disk_indirect_block);
  return physical_sector;

  // /* logical_sector is too large */
  // return NOT_A_SECTOR;
}

/* Initializes the inode module. */
void
inode_init (void) 
{
  list_init (&open_inodes);
  template_init();
}


/* Initializes an inode with LENGTH bytes of data and
   writes the new inode to sector SECTOR on the file system
   device.
   Returns true if successful.
   Returns false if memory or disk allocation fails. */
bool
inode_create (block_sector_t sector, off_t length, bool isdir)
{
  struct inode_disk *disk_inode = NULL;

  ASSERT (length >= 0);

  /* If this assertion fails, the inode structure is not exactly
     one sector in size, and you should fix that. */
  ASSERT (sizeof *disk_inode == BLOCK_SECTOR_SIZE);

  disk_inode = calloc (1, sizeof *disk_inode);
  if (disk_inode == NULL)
    return false;
  /* init inode */
  memcpy(disk_inode, &template_disk_inode, sizeof *disk_inode);
  
  size_t sectors = bytes_to_sectors (length);
  disk_inode->length = length;
  disk_inode->magic = INODE_MAGIC;
  disk_inode->isdir = isdir;
  /* creates all nodes by inode_seek */
  for(size_t i = 0; i < sectors; i++){
      block_sector_t physical_sector = inode_seek(disk_inode, i);
      if(physical_sector == NOT_A_SECTOR){
          free (disk_inode);
          return false;
        }
    }
  cache_write (sector, disk_inode);
  free (disk_inode);
  return true;
}

/* Reads an inode from SECTOR
   and returns a `struct inode' that contains it.
   Returns a null pointer if memory allocation fails. */
struct inode *
inode_open (block_sector_t sector)
{
  sector %= FD_GROW_MAGIC;
  struct list_elem *e;
  struct inode *inode;

  /* Check whether this inode is already open. */
  for (e = list_begin (&open_inodes); e != list_end (&open_inodes);
       e = list_next (e)) 
    {
      inode = list_entry (e, struct inode, elem);
      if (inode->sector == sector) 
        {
          inode_reopen (inode);
          return inode; 
        }
    }

  /* Allocate memory. */
  inode = malloc (sizeof *inode);
  if (inode == NULL)
    return NULL;

  /* Initialize. */
  list_push_front (&open_inodes, &inode->elem);
  inode->sector = sector;
  inode->open_cnt = 1;
  inode->deny_write_cnt = 0;
  inode->removed = false;
  cache_read (inode->sector, &inode->data);
  return inode;
}

/* Reopens and returns INODE. */
struct inode *
inode_reopen (struct inode *inode)
{
  if (inode != NULL)
    inode->open_cnt++;
  return inode;
}

/* Returns INODE's inode number. */
block_sector_t
inode_get_inumber (const struct inode *inode)
{
  return inode->sector;
}

/* Closes INODE and writes it to disk.
   If this was the last reference to INODE, frees its memory.
   If INODE was also a removed inode, frees its blocks. */
void
inode_close (struct inode *inode) 
{
  /* Ignore null pointer. */
  if (inode == NULL)
    return;
  //TODO: closes 
  /* Release resources if this was the last opener. */
  if (--inode->open_cnt == 0){
      /* Remove from inode list and release lock. */
      list_remove (&inode->elem);
 
      /* Deallocate blocks if removed. */
      if (inode->removed) {
        /* free all direct blocks */
        for(int i = 0; i < DIRECT_BLOCK_SIZE; i++){
          if(inode->data.direct_blocks[i] != NOT_A_SECTOR)
            free_map_release (inode->data.direct_blocks[i], 1);
        }
        struct double_indirect_block_disk_t * double_indirect_block_disk = NULL;
        struct indirect_block_disk_t * indirect_block_disk = NULL;

        double_indirect_block_disk = malloc(sizeof *double_indirect_block_disk);
        indirect_block_disk = malloc(sizeof *indirect_block_disk);

        if(double_indirect_block_disk == NULL || indirect_block_disk == NULL){
          free(double_indirect_block_disk);
          free(indirect_block_disk);
          PANIC("memory allocation failed while freeing filesystem blocks");
        }

        /* free all indirect blocks */
        if (inode->data.indirect_block != NOT_A_SECTOR){

          cache_read(inode->data.indirect_block, indirect_block_disk);
          for(int i = 0; i < INDIRECT_BLOCK_SIZE; i++){
            block_sector_t physical_sector = indirect_block_disk->direct_blocks[i];
            if(physical_sector != NOT_A_SECTOR)
              free_map_release (physical_sector, 1);
          }
          free_map_release (inode->data.indirect_block, 1);
        }

        /* free all double indirect blocks */
        for(int _i = 0; _i < DOUBLE_INDIRECT_BLOCK_IN_INODE_SIZE; ++_i){
          if (inode->data.double_indirect_block[_i] != NOT_A_SECTOR){
            cache_read(inode->data.double_indirect_block[_i], double_indirect_block_disk);

            /* free all indirect blocks inside the double indirect block*/
            for(int i = 0; i < DOUBLE_INDIRECT_BLOCK_SIZE; i++){
              block_sector_t indirect_block = double_indirect_block_disk->indirect_block_disk[i];
              if (indirect_block == NOT_A_SECTOR)
                continue;

              cache_read(indirect_block, indirect_block_disk);

              /* free all direct blocks inside the indirect block */
              for(int i = 0; i < INDIRECT_BLOCK_SIZE; ++i){
                block_sector_t physical_sector = indirect_block_disk->direct_blocks[i];
                if(physical_sector != NOT_A_SECTOR)
                  free_map_release (physical_sector, 1);
              }
              free_map_release(indirect_block, 1);

            }

            free_map_release (inode->data.double_indirect_block[_i], 1);
          }
        }

        free(double_indirect_block_disk);
        free(indirect_block_disk);
        free_map_release (inode->sector, 1);
      }
      else { /* not removed, excute write back */
        cache_write(inode->sector, &inode->data);
      }

      free (inode); 
    }
}

/* Marks INODE to be deleted when it is closed by the last caller who
   has it open. */
void
inode_remove (struct inode *inode) 
{
  ASSERT (inode != NULL);
  inode->removed = true;
}

/* Reads SIZE bytes from INODE into BUFFER, starting at position OFFSET.
   Returns the number of bytes actually read, which may be less
   than SIZE if an error occurs or end of file is reached. */
off_t
inode_read_at (struct inode *inode, void *buffer_, off_t size, off_t offset) 
{
  ASSERT(inode != NULL);
  uint8_t *buffer = buffer_;
  off_t bytes_read = 0;
  uint8_t *bounce = malloc (BLOCK_SECTOR_SIZE);
  if(bounce == NULL){
    return 0;
  }
  if (size == 0){
    free(bounce);
    return 0;
  }
  if (offset >= inode->data.length){
    free(bounce);
    return 0;
  }

  if (offset + size > inode->data.length)
    size = inode->data.length - offset;
  /* read the first sector */
  block_sector_t first = offset / BLOCK_SECTOR_SIZE;
  block_sector_t sector_idx = inode_seek(&inode->data, first);
  if(sector_idx == NOT_A_SECTOR){
    free (bounce);
    return 0;
  }
  cache_read(sector_idx, bounce);
  int sector_ofs = offset % BLOCK_SECTOR_SIZE;
  int sector_left = BLOCK_SECTOR_SIZE - sector_ofs;
  int chunk_size = size < sector_left ? size : sector_left;
  int logical_sector_cnt = 1;
  memcpy(buffer + bytes_read, bounce + sector_ofs, chunk_size);
  bytes_read += chunk_size;
  /* read the rest sectors */
  while (bytes_read < size){
      block_sector_t sector_idx = inode_seek(&inode->data, first + logical_sector_cnt);
      if(sector_idx == NOT_A_SECTOR){
        free (bounce);
        return 0;
      }
      cache_read(sector_idx, bounce);
      int chunk_size = size - bytes_read < BLOCK_SECTOR_SIZE ? size - bytes_read : BLOCK_SECTOR_SIZE;
      memcpy(buffer + bytes_read, bounce, chunk_size);
      bytes_read += chunk_size;
      logical_sector_cnt ++;
    }
  free (bounce);
  return bytes_read;
}

/* Writes SIZE bytes from BUFFER into INODE, starting at OFFSET.
   Returns the number of bytes actually written, which may be
   less than SIZE if end of file is reached or an error occurs.
   (Normally a write at end of file would extend the inode, but
   growth is not yet implemented.) */
off_t
inode_write_at (struct inode *inode, const void *buffer_, off_t size,
                off_t offset) 
{
  const uint8_t *buffer = buffer_;
  off_t bytes_written = 0;
  uint8_t *bounce = NULL;

  if (inode->deny_write_cnt)
    return 0;
  if (size == 0)
    return 0;
  bounce = malloc (BLOCK_SECTOR_SIZE);
  /* write the first sector */
  block_sector_t first = offset / BLOCK_SECTOR_SIZE;
  block_sector_t sector_idx = inode_seek(&inode->data, first);
  if(sector_idx == NOT_A_SECTOR){
    free (bounce);
    return 0;
  }
  cache_read(sector_idx, bounce);
  int sector_ofs = offset % BLOCK_SECTOR_SIZE;
  int sector_left = BLOCK_SECTOR_SIZE - sector_ofs;
  int chunk_size = size < sector_left ? size : sector_left;
  memcpy(bounce + sector_ofs, buffer + bytes_written, chunk_size);
  cache_write (sector_idx, bounce);
  bytes_written += chunk_size;
  /* write the rest sectors */
  int logical_sector_cnt = 0;
  ++ logical_sector_cnt;

  while (bytes_written < size){
    block_sector_t sector_idx = inode_seek(&inode->data, first + logical_sector_cnt);
    if(sector_idx == NOT_A_SECTOR)
      break;
    int chunk_size = size - bytes_written < BLOCK_SECTOR_SIZE ? size - bytes_written : BLOCK_SECTOR_SIZE;
    if (chunk_size > 0){
        if (chunk_size == BLOCK_SECTOR_SIZE) {
            /* Write full sector directly to disk. */
            cache_write (sector_idx, buffer + bytes_written);
          }
        else {
            cache_read (sector_idx, bounce);
            memcpy (bounce, buffer + bytes_written, chunk_size);
            cache_write (sector_idx, bounce);
          }
        /* Advance. */
        bytes_written += chunk_size;
      }
    logical_sector_cnt ++;
  }

  free (bounce);
  /* update inode length */
  if (offset + bytes_written > inode->data.length)
      inode->data.length = offset + bytes_written;
  /* write disk_inode back*/
  cache_write(inode->sector, &inode->data);
  return bytes_written;
}

/* Disables writes to INODE.
   May be called at most once per inode opener. */
void
inode_deny_write (struct inode *inode) 
{
  inode->deny_write_cnt++;
  ASSERT (inode->deny_write_cnt <= inode->open_cnt);
}

/* Re-enables writes to INODE.
   Must be called once by each inode opener who has called
   inode_deny_write() on the inode, before closing the inode. */
void
inode_allow_write (struct inode *inode) 
{
  ASSERT (inode->deny_write_cnt > 0);
  ASSERT (inode->deny_write_cnt <= inode->open_cnt);
  inode->deny_write_cnt--;
}

/* Returns the length, in bytes, of INODE's data. */
off_t
inode_length (const struct inode *inode)
{
  ASSERT(inode->data.magic == INODE_MAGIC);
  return inode->data.length;
}

bool inode_is_dir(const struct inode *inode){
  /* assert inode is not null and indeed an inode*/
  ASSERT(inode != NULL);
  ASSERT(inode->data.magic == INODE_MAGIC);
  return inode->data.isdir;
}

bool inode_is_file(const struct inode * inode){
  /* assert inode is not null and indeed an inode*/
  ASSERT(inode != NULL);
  ASSERT(inode->data.magic == INODE_MAGIC);
  return !inode->data.isdir;
}

bool inode_is_removed(const struct inode *inode){
  ASSERT(inode != NULL);
  ASSERT(inode->data.magic == INODE_MAGIC);
  return inode->removed;
}

bool inode_is_opened(const struct inode *inode){
  ASSERT(inode != NULL);
  ASSERT(inode->data.magic == INODE_MAGIC);
  return inode->open_cnt > 0;
}

