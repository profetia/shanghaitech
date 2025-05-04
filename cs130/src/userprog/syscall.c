#include "userprog/syscall.h"
#include "devices/input.h"
#include "devices/shutdown.h"
#include "filesys/directory.h"
#include "filesys/file.h"
#include "filesys/filesys.h"
#include "filesys/inode.h"
#include "stdbool.h"
#include "stddef.h"
#include "threads/interrupt.h"
#include "threads/thread.h"
#include "threads/vaddr.h"
#include "userprog/pagedir.h"
#include "userprog/process.h"
#include <stdio.h>
#include <syscall-nr.h>
#include <tanc.h>
#ifdef VM
#include "vm/page.h"
#endif

static void syscall_handler (struct intr_frame *);

static void syscall_halt (void);
static void syscall_exit (int status);
static tid_t syscall_exec (const char *cmd_line);
static int syscall_wait (tid_t pid);
static bool syscall_create (const char *file, off_t initial_size);
static bool syscall_remove (const char *fileOrDir);
static int syscall_open (const char *fileOrDir);
static int syscall_filesize (int fd);
static int syscall_read (int fd, void *buffer, unsigned size);
static int syscall_write (int fd, const void *buffer, unsigned size);
static void syscall_seek (int fd, unsigned position);
static unsigned syscall_tell (int fd);
static void syscall_close (int fd);
static bool syscall_chdir (const char *dir);
static bool syscall_mkdir (const char *dir);
static bool syscall_readdir (int fd, char *name);
static bool syscall_isdir (int fd);
static int syscall_inumber (int fd);

#ifdef VM
static int syscall_mmap (int fd, void *addr);
static void syscall_munmap (int mapid);
#endif

static bool is_valid_vaddr (const void *vaddr, bool write);
static bool is_valid_vrange (const void *vaddr, unsigned size, bool write);
static bool is_valid_word (const void *vaddr, bool write);
static bool is_valid_string (const char *str, bool write);

void
syscall_init (void)
{
  intr_register_int (0x30, 3, INTR_ON, syscall_handler, "syscall");
}

static void
syscall_handler (struct intr_frame *f)
{
  if (!is_valid_word (f->esp, false))
    {
      syscall_exit (-1);
    }

  int sys_code = *(int *)f->esp;

  switch (sys_code)
    {
    case SYS_HALT:
      syscall_halt ();
      break;
    case SYS_EXIT:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      syscall_exit (*(int *)(f->esp + 4));
      break;
    case SYS_EXEC:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_exec (*(char **)(f->esp + 4));
      break;
    case SYS_WAIT:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_wait (*(tid_t *)(f->esp + 4));
      break;
    case SYS_CREATE:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false))
        syscall_exit (-1);
      f->eax = syscall_create (*(char **)(f->esp + 4), *(off_t *)(f->esp + 8));
      break;
    case SYS_REMOVE:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_remove (*(char **)(f->esp + 4));
      break;
    case SYS_OPEN:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_open (*(char **)(f->esp + 4));
      break;
    case SYS_FILESIZE:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_filesize (*(int *)(f->esp + 4));
      break;
    case SYS_READ:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false)
          || !is_valid_word (f->esp + 12, false))
        syscall_exit (-1);
      f->eax = syscall_read (*(int *)(f->esp + 4), *(void **)(f->esp + 8),
                             *(unsigned *)(f->esp + 12));
      break;
    case SYS_WRITE:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false)
          || !is_valid_word (f->esp + 12, false))
        syscall_exit (-1);
      f->eax = syscall_write (*(int *)(f->esp + 4), *(void **)(f->esp + 8),
                              *(unsigned *)(f->esp + 12));
      break;
    case SYS_SEEK:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false))
        syscall_exit (-1);
      syscall_seek (*(int *)(f->esp + 4), *(unsigned *)(f->esp + 8));
      break;
    case SYS_TELL:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_tell (*(int *)(f->esp + 4));
      break;
    case SYS_CLOSE:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      syscall_close (*(int *)(f->esp + 4));
      break;
#ifdef VM
    case SYS_MMAP:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false))
        syscall_exit (-1);
      f->eax = syscall_mmap (*(int *)(f->esp + 4), *(void **)(f->esp + 8));
      break;
    case SYS_MUNMAP:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      syscall_munmap (*(int *)(f->esp + 4));
      break;
#endif
    case SYS_CHDIR:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_chdir (*(char **)(f->esp + 4));
      break;
    case SYS_MKDIR:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_mkdir (*(char **)(f->esp + 4));
      break;
    case SYS_READDIR:
      if (!is_valid_word (f->esp + 4, false)
          || !is_valid_word (f->esp + 8, false))
        syscall_exit (-1);
      f->eax = syscall_readdir (*(int *)(f->esp + 4), *(char **)(f->esp + 8));
      break;
    case SYS_INUMBER:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_inumber (*(int *)(f->esp + 4));
      break;
    case SYS_ISDIR:
      if (!is_valid_word (f->esp + 4, false))
        syscall_exit (-1);
      f->eax = syscall_isdir (*(int *)(f->esp + 4));
      break;


    default:
      syscall_exit (-1);
    }
}

static void
syscall_halt (void)
{
  shutdown_power_off ();
}

static void
syscall_exit (int status)
{
  thread_current ()->exit_status = status;
  thread_exit ();
}

static tid_t
syscall_exec (const char *cmd_line)
{
  if (!is_valid_string (cmd_line, false))
    syscall_exit (-1);
  return process_execute (cmd_line, thread_current ()->cwd_fd);
}

static int
syscall_wait (tid_t pid)
{
  return process_wait (pid);
}

static bool
syscall_create (const char *file, off_t initial_size)
{
  if (!is_valid_string (file, false))
    syscall_exit (-1);
  lock_acquire (&fs_lock);
  bool success
      = filesys_create (file, initial_size, thread_current ()->cwd_fd, false);
  int parent_fd = NOT_A_FD;
  path_seek (file, thread_current ()->cwd_fd, &parent_fd);
  lock_release (&fs_lock);
  return success;
}

static bool
syscall_remove (const char *fileOrDir)
{
  if (!is_valid_string (fileOrDir, false))
    syscall_exit (-1);
  lock_acquire (&fs_lock);
  bool success = filesys_remove (fileOrDir, thread_current ()->cwd_fd);
  lock_release (&fs_lock);
  return success;
}

static int
syscall_open (const char *file)
{
  if (!is_valid_string (file, false))
    syscall_exit (-1);

  lock_acquire (&fs_lock);
  // struct file *f = filesys_open(file,thread_current()->cwd_fd);
  int parent_fd = NOT_A_FD;
  struct inode *inode
      = path_seek (file, thread_current ()->cwd_fd, &parent_fd);
  lock_release (&fs_lock);
  if (inode == NULL)
    {
      return -1;
    }


  if (inode_is_dir (inode))
    {
      struct file *file = file_open (inode);
      return process_add_file (file);
    }
  if (inode_is_file (inode))
    {
      struct file *file = file_open (inode);
      return process_add_file (file);
    }
  /* other undefined cases? */
  return -1;
}

static int
syscall_filesize (int fd)
{
  struct file *f = process_get_file (fd);
  if (f == NULL)
    {
      syscall_exit (-1);
    }
  lock_acquire (&fs_lock);
  int size = file_length (f);
  lock_release (&fs_lock);
  return size;
}

static int
syscall_read (int fd, void *buffer, unsigned size)
{
  if (size == 0)
    return 0;

  if (!is_valid_vrange (buffer, size, true))
    syscall_exit (-1);

  if (fd == STDIN_FILENO)
    {
      uint8_t *buf = (uint8_t *)buffer;
      for (unsigned i = 0; i < size; i++)
        {
          buf[i] = input_getc ();
        }
      return (int)size;
    }

  struct file *f = process_get_file (fd);
  if (f == NULL || file_is_dir (f))
    syscall_exit (-1);

  lock_acquire (&fs_lock);
  int bytes_read = file_read (f, buffer, (off_t)size);
  lock_release (&fs_lock);
  return bytes_read;
}

static int
syscall_write (int fd, const void *buffer, unsigned size)
{
  if (size == 0)
    return 0;

  if (!is_valid_vrange (buffer, size, false))
    syscall_exit (-1);

  if (fd == STDOUT_FILENO)
    {
      putbuf (buffer, size);
      return (int)size;
    }

  struct file *f = process_get_file (fd);
  if (f == NULL || file_is_dir (f))
    syscall_exit (-1);

  lock_acquire (&fs_lock);
  int bytes_written = file_write (f, buffer, (off_t)size);
  lock_release (&fs_lock);
  return bytes_written;
}

static void
syscall_seek (int fd, unsigned position)
{
  struct file *f = process_get_file (fd);
  if (f == NULL)
    syscall_exit (-1);
  lock_acquire (&fs_lock);
  file_seek (f, (off_t)position);
  lock_release (&fs_lock);
}

static unsigned
syscall_tell (int fd)
{
  struct file *f = process_get_file (fd);
  if (f == NULL)
    syscall_exit (-1);
  lock_acquire (&fs_lock);
  unsigned position = file_tell (f);
  lock_release (&fs_lock);
  return position;
}

static void
syscall_close (int fd)
{
  struct file *f = process_get_file (fd);
  if (f == NULL)
    syscall_exit (-1);

  process_close_file (fd);
}

#ifdef VM
static int
syscall_mmap (int fd, void *addr)
{
  if (addr == NULL || pg_ofs (addr) != 0)
    return MAPID_ERROR;

  struct file *f = process_get_file (fd);
  if (f == NULL)
    return MAPID_ERROR;

  lock_acquire (&fs_lock);
  f = file_reopen (f);
  lock_release (&fs_lock);
  if (f == NULL)
    return MAPID_ERROR;

  return process_add_mmap (f, addr);
}

static void
syscall_munmap (int mapid)
{
  process_remove_mmap (mapid);
}
#endif

/* Returns true if the given virtual address is valid,
   which is to say that it is not NULL, it is in user
   memory, and it is mapped to a page in the current
   process's page table. Returns false otherwise. */
static bool
is_valid_vaddr (const void *vaddr, bool write UNUSED)
{
#ifdef VM
  if (vaddr == NULL || !is_user_vaddr (vaddr))
    return false;

  struct sup_page_table_entry *entry
      = page_find (&thread_current ()->sup_page_table, vaddr);

  if (entry == NULL)
    return false;
  if (write && !entry->writable)
    return false;
  if (entry->location == PAGE_LOC_ERROR)
    return false;

  return true;
#else
  return vaddr != NULL && is_user_vaddr (vaddr)
         && pagedir_get_page (thread_current ()->pagedir, vaddr) != NULL;
#endif
}

/* Returns true if the given virtual address range is valid,
   false otherwise. */
static bool
is_valid_vrange (const void *vaddr, unsigned size, bool write)
{
  const void *end = vaddr + size - 1;
  if (end < vaddr)
    {
      return false;
    }
  return is_valid_vaddr (vaddr, write) && is_valid_vaddr (end, write);
}

static bool
is_valid_word (const void *vaddr, bool write)
{
  return is_valid_vrange (vaddr, sizeof (int), write);
}

/* Returns true if the given string is valid, false
   otherwise. */
static bool
is_valid_string (const char *str, bool write)
{
  while (is_valid_vaddr (str, write))
    {
      if (*str == '\0')
        {
          return true;
        }
      str++;
    }
  return false;
}

static bool
syscall_chdir (const char *dir)
{
  lock_acquire(&fs_lock);
  struct inode *inode = path_seek (dir, thread_current ()->cwd_fd, NULL);
  lock_release(&fs_lock);
  if (inode == NULL)
    return false;
  if (!inode_is_dir (inode))
    {
      inode_close (inode);
      return false;
    }
  int target_fd = inode_get_inumber (inode);
  inode_close (inode);
  thread_current ()->cwd_fd = target_fd;
  return true;
}

static bool
syscall_mkdir (const char *dir)
{
  lock_acquire (&fs_lock);
  bool success = filesys_create (dir, 0, thread_current ()->cwd_fd, true);
  lock_release (&fs_lock);
  return success;
}

static bool
syscall_readdir (int fd, char *name)
{
  struct file *file = process_get_file (fd);
  if(file == NULL){
    return false;
  }
  if(!file_is_dir(file)){
    return false;
  }
  struct dir *dir = file_get_dir(file);
  if (dir == NULL){
    return false;
  }
  lock_acquire(&fs_lock);
  bool success = dir_readdir (dir, name); //TODO where is name allocated?
  lock_release(&fs_lock);
  return success;
}
static bool
syscall_isdir (int fd)
{
  struct file *file = process_get_file (fd);
  if (file == NULL)
    return false;
  struct inode *inode = file_get_inode (file);
  if (inode == NULL)
    return false;
  return inode_is_dir (inode);
}

static int
syscall_inumber (int fd)
{
  return fd % FD_GROW_MAGIC;
}