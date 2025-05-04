#include "filesys/filesys.h"
#include <debug.h>
#include <stdio.h>
#include <string.h>
#include "filesys/file.h"
#include "filesys/free-map.h"
#include "filesys/inode.h"
#include "filesys/directory.h"
#include "threads/thread.h"
#include "threads/malloc.h"
#include "filesys/cache.h"

/* Partition that contains the file system. */
struct block *fs_device;

/* this is a function for hints. this function won't be called.*/
block_sector_t fd_to_sector(int fd){
  /* defined so. in this filesys the inode sector number is the filesystem descriptor */
  return fd;
}

static void do_format (void);

/* Initializes the file system module.
   If FORMAT is true, reformats the file system. */
void
filesys_init (bool format) 
{
  fs_device = block_get_role (BLOCK_FILESYS);
  if (fs_device == NULL)
    PANIC ("No file system device found, can't initialize file system.");

  inode_init ();
  free_map_init ();
  if (format) 
    do_format ();

  free_map_open ();
}

/* Shuts down the file system module, writing any unwritten data
   to disk. */
void
filesys_done (void) 
{
  cache_write_behind(true);
  cache_flush();  
  free_map_close ();
}

/* Creates a file named NAME with the given INITIAL_SIZE.
   Returns true if successful, false otherwise.
   Fails if a file named NAME already exists,
   or if internal memory allocation fails. */
bool
filesys_create (const char *name, off_t initial_size, int cwd_fd,bool isdir)
 {
  block_sector_t inode_sector = 0;
  int parent_fd = NOT_A_FD;
  struct inode * inode = path_seek(name, cwd_fd, &parent_fd);
  //if the file/dir already exists, return false
  if(inode != NULL){
    inode_close(inode);
    return false;
  }
  //if the parent directory does not exist, return false
  if(parent_fd == NOT_A_FD){
    return false;
  }
  struct dir *dir = dir_open(inode_open(parent_fd));
  bool success = false;
  if(!isdir){
    char * copy_name = malloc(strlen(name)+1);
    char * last_token = NULL;
    strlcpy(copy_name,name,strlen(name)+1);    
    success = (get_last_token(copy_name, &last_token) 
                    && dir != NULL
                    && free_map_allocate (1, &inode_sector)
                    && inode_create (inode_sector, initial_size, false)
                    && dir_add (dir, last_token, inode_sector));
    free(copy_name);
  }else{
    success = (dir != NULL)
                    && free_map_allocate (1, &inode_sector)
                    && dir_create (inode_sector, 16, parent_fd);
    if(success){
      char * copy_name = malloc(strlen(name)+1);
      char * last_token = NULL;
      strlcpy(copy_name,name,strlen(name)+1);
      bool local_success = get_last_token(copy_name, &last_token) 
                           && dir_create(inode_sector, 16, parent_fd) 
                           && dir_add (dir, last_token, inode_sector);
      free(copy_name);
      success = success && local_success;
    }
  }
  if (!success && inode_sector != (unsigned) NOT_A_SECTOR) 
    free_map_release (inode_sector, 1);
  dir_close (dir);
  return success;
}


/* Opens the file with the given NAME.
   Returns the new file if successful or a null pointer
   otherwise.
   Fails if no file named NAME exists,
   or if an internal memory allocation fails. */
struct file *
filesys_open (const char *_name, int cwd_fd)
{
  // struct dir *dir = dir_open(inode_open(cwd_fd));
  // if(dir == NULL)
  //   return NULL;
  // struct inode *inode = NULL;

  // if (dir != NULL)
  //   dir_lookup (dir, _name, &inode);
  // dir_close (dir);
  struct inode *inode2 = path_seek(_name, cwd_fd, NULL);
  // inode_close(inode2);

  return file_open (inode2);
  
  /* if _name is start with '/' then it is an absolute path */
  // struct inode *inode = path_seek(_name, cwd_fd);
  /* the node is already opened in path_seek */
  // return file_open(inode);
}

/* Deletes the file named NAME.
   Returns true if successful, false on failure.
   Fails if no file named NAME exists,
   or if an internal memory allocation fails. */
bool
filesys_remove (const char *name, int cwd_fd) 
{
  // struct dir *dir = dir_open(inode_open(cwd_fd));
  // if(dir == NULL) return false;
  // bool success = dir != NULL && dir_remove (dir, name);
  // dir_close (dir); 
  int parent_fd = NOT_A_FD;
  struct inode *inode = path_seek(name, cwd_fd, &parent_fd);
  if(inode == NULL)
    return false;
  if(inode_is_dir(inode)){
    /* if the inode is a directory, then we need to check whether it is empty */
    struct dir * dir_local = dir_open(inode);
    if(!dir_is_empty(dir_local)){
      /* if the directory is not empty, then we return false */
      dir_close(dir_local);
      return false;
    }
    free(dir_local);
  }
  if(parent_fd == NOT_A_FD){
    inode_close(inode);
    return false;
  }
  struct dir * dir = dir_open(inode_open(parent_fd));
  if(dir == NULL){
    inode_close(inode);
    dir_close(dir);    
    return false;
  }
  inode_close(inode);
  //get the last token of name 
  char * copy_name = malloc(strlen(name)+1);
  char * last_token = NULL;
  strlcpy(copy_name, name, strlen(name)+1);
  bool success = get_last_token(copy_name, &last_token) 
                       && dir_remove (dir, last_token);
  free(copy_name);
  dir_close(dir);
  return success;
  // return success;
}

/* Formats the file system. */
static void
do_format (void)
{
  printf ("Formatting file system...");
  free_map_create ();
  if (!dir_create (ROOT_DIR_SECTOR, 16, ROOT_DIR_SECTOR))
    PANIC ("root directory creation failed");
  free_map_close ();
  printf ("done.\n");
}
