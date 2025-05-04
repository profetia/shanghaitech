#ifndef FILESYS_FILESYS_H
#define FILESYS_FILESYS_H

#include <stdbool.h>
#include "filesys/off_t.h"
#include "devices/block.h"

/* Sectors of system file inodes. */
#define FREE_MAP_SECTOR 0       /* Free map file inode sector. */
#define ROOT_DIR_SECTOR 2       /* Root directory file inode sector. */
#define ROOT_DIR_FD ROOT_DIR_SECTOR /* Root directory file descriptor. */

/* Block device that contains the file system. */
extern struct block *fs_device;

block_sector_t fd_to_sector (int fd);
void filesys_init (bool format);
void filesys_done (void);
bool filesys_create (const char *name, off_t initial_size,int cwd_fd,bool isdir);
struct file *filesys_open (const char *name, int cwd_fd);
bool filesys_remove (const char *name, int cwd_fd);

#endif /* filesys/filesys.h */
