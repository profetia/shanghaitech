#include "userprog/process.h"
#include <debug.h>
#include <inttypes.h>
#include <round.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tanc.h>
#include <list.h>
#include "filesys/inode.h"
#include "userprog/gdt.h"
#include "userprog/pagedir.h"
#include "userprog/tss.h"
#include "filesys/directory.h"
#include "filesys/file.h"
#include "filesys/filesys.h"
#include "threads/flags.h"
#include "threads/init.h"
#include "threads/interrupt.h"
#include "threads/malloc.h"
#include "threads/palloc.h"
#include "threads/thread.h"
#include "threads/vaddr.h"

#ifdef VM
#include "vm/page.h"
#endif

static thread_func start_process NO_RETURN;
static bool load (struct list* arg_list, void (**eip) (void), void **esp);

static struct list* parse_args(const char* file_name);
static void cleanup_args(struct list* arg_list);

struct lock fs_lock;

/* Initializes the process system */
void
process_init (void) 
{
  lock_init(&fs_lock);
}

struct arg_elem 
  {
    char* arg;
    char* addr;
    struct list_elem elem;
  };

struct child_elem 
  {
    struct thread* child;
    int exit_status;
    struct semaphore sema;
    tid_t pid;
    struct list_elem elem;    
  };

struct start_process_args
  {
    struct list* arg_list;
    struct thread* parent;
    struct child_elem* child;
  };

/* Starts a new thread running a user program loaded from
   FILENAME.  The new thread may be scheduled (and may even exit)
   before process_execute() returns.  Returns the new process's
   thread id, or TID_ERROR if the thread cannot be created. */
tid_t
process_execute (const char *file_name, int cwd_fd) 
{
  struct list* arg_list;
  tid_t tid;

  /* Make a copy of FILE_NAME.
     Otherwise there's a race between the caller and load(). */
  arg_list = parse_args(file_name);
  if (arg_list == NULL)
    return TID_ERROR;     

  // Get the program name from the arg_list
  char* exec_name = list_entry(
      list_front(arg_list), struct arg_elem, elem)->arg;

  struct thread* cur = thread_current();

  struct start_process_args* init_args = malloc(sizeof(struct start_process_args));
  if (init_args == NULL)
    {
      cleanup_args(arg_list);
      return TID_ERROR;
    }
  init_args->arg_list = arg_list;
  init_args->parent = cur;

  struct child_elem* child = malloc(sizeof(struct child_elem));
  if (child == NULL)
    {
      cleanup_args(arg_list);
      free(init_args);
      return TID_ERROR;
    }
  init_args->child = child;

  sema_init(&child->sema, 0);
  child->pid = TID_ERROR;
  child->exit_status = -1;
  child->child = NULL;

  lock_acquire (&cur->child_lock);
  list_push_back(&cur->child_list, &child->elem);
  lock_release (&cur->child_lock);  

  /* Create a new thread to execute FILE_NAME. */
  tid = thread_create (exec_name, PRI_DEFAULT, start_process, init_args, cwd_fd);
  if (tid == TID_ERROR)
    {
      cleanup_args(arg_list);
      free(child);
      free(init_args);
      return TID_ERROR;
    }
    
  sema_down(&child->sema);
  if (child->pid == TID_ERROR)
    {
      lock_acquire (&cur->child_lock);
      list_remove(&child->elem);
      lock_release (&cur->child_lock);
      free(child);
      return TID_ERROR;
    }

  return tid;
}

/* Parse the arguments into a list of arg_elem structs. If the
   file_name is "echo x y z", then the list will contain the
   following elements: "echo", "x", "y", "z". 
   The caller is responsible for freeing the list. */
static struct list* 
parse_args(const char* file_name)
{
  size_t fn_len = strlen(file_name);
  char* fn_copy = malloc(fn_len + 1);
  if (fn_copy == NULL)
    return NULL;

  strlcpy(fn_copy, file_name, fn_len + 1);

  struct list* arg_list = malloc(sizeof(struct list));
  if (arg_list == NULL)
    {
      free(fn_copy);
      return NULL;
    }

  list_init(arg_list);

  char* token, *rest;
  for (token = strtok_r(fn_copy, " ", &rest); token != NULL; token = strtok_r(NULL, " ", &rest))
    {
      struct arg_elem* arg = malloc(sizeof(struct arg_elem));
      if (arg == NULL)
        {
          cleanup_args(arg_list);
          free(fn_copy);
          return NULL;
        }
      memset(arg, 0, sizeof(struct arg_elem));

      size_t token_len = strlen(token);

      arg->arg = malloc(token_len + 1);
      if (arg->arg == NULL)
        {
          free(arg);
          cleanup_args(arg_list);
          free(fn_copy);
          return NULL;
        }
      strlcpy(arg->arg, token, token_len + 1);
      
      list_push_back(arg_list, &arg->elem);
    }

  free(fn_copy);
  return arg_list;
}

/* Free the memory allocated by parse_args. */
static void
cleanup_args(struct list* arg_list)
{
  struct list_elem* e;
  while (!list_empty(arg_list))
    {
      e = list_pop_front(arg_list);
      struct arg_elem* arg = list_entry(e, struct arg_elem, elem);
      free(arg->arg);
      free(arg);
    }
  free(arg_list);
}

/* A thread function that loads a user process and starts it
   running. */
static void
start_process (void *init_args_)
{
  struct start_process_args* init_args = init_args_;
  struct list* arg_list = init_args->arg_list;
  struct child_elem* child = init_args->child;
  struct thread* parent = init_args->parent;
  struct intr_frame if_;
  bool success;

#ifdef VM
  sup_page_table_init (&thread_current()->sup_page_table);
  list_init (&thread_current()->mmap_list);
  thread_current()->next_mapid = 0;
#endif  

  /* Initialize interrupt frame and load executable. */
  memset (&if_, 0, sizeof if_);
  if_.gs = if_.fs = if_.es = if_.ds = if_.ss = SEL_UDSEG;
  if_.cs = SEL_UCSEG;
  if_.eflags = FLAG_IF | FLAG_MBS;
  success = load (arg_list, &if_.eip, &if_.esp);

  /* If load failed, quit. */
  cleanup_args (arg_list);
  free (init_args);
  if (!success) 
    {    
      sema_up(&child->sema);    
      thread_exit ();  
    }
  else
    {
      struct thread* cur = thread_current();
      cur->parent = parent;
      child->pid = cur->tid;
      child->child = cur;
      sema_up(&child->sema);
    }

  /* Start the user process by simulating a return from an
     interrupt, implemented by intr_exit (in
     threads/intr-stubs.S).  Because intr_exit takes all of its
     arguments on the stack in the form of a `struct intr_frame',
     we just point the stack pointer (%esp) to our stack frame
     and jump to it. */
  asm volatile ("movl %0, %%esp; jmp intr_exit" : : "g" (&if_) : "memory");
  NOT_REACHED ();
}

static bool process_child_pred (const struct list_elem*, void* aux);
static bool process_pid_pred (const struct list_elem*, void* aux);

static bool
process_pid_pred (const struct list_elem* e, void* aux)
{
  struct child_elem* child = list_entry(e, struct child_elem, elem);
  return child->pid == *(tid_t*)aux;
}

/* Waits for thread TID to die and returns its exit status.  If
   it was terminated by the kernel (i.e. killed due to an
   exception), returns -1.  If TID is invalid or if it was not a
   child of the calling process, or if process_wait() has already
   been successfully called for the given TID, returns -1
   immediately, without waiting.

   This function will be implemented in problem 2-2.  For now, it
   does nothing. */
int
process_wait (tid_t child_tid) 
{
  struct thread* cur = thread_current();

  lock_acquire(&cur->child_lock);
  struct list_elem* e = list_find_if(
      &cur->child_list, process_pid_pred, &child_tid);
  lock_release(&cur->child_lock);
  if (e == NULL)
    return -1;

  struct child_elem* child = list_entry(e, struct child_elem, elem);

  ASSERT (child->pid != TID_ERROR);
  if (child->child != NULL)
    sema_down(&child->sema);    
  
  int exit_status = child->exit_status;
  lock_acquire(&cur->child_lock);
  list_remove(&child->elem);
  lock_release(&cur->child_lock);
  free(child);
  return exit_status;
}

struct file_elem
  {
    struct file* file;
    int fd;
    struct list_elem elem;
  }; 

static bool 
process_child_pred (const struct list_elem* e, void* aux)
{
  struct child_elem* child = list_entry (e, struct child_elem, elem);
  return child->child == (struct thread*)aux;
}

/* Free the current process's resources. */
void
process_exit (void)
{
  struct thread *cur = thread_current ();
  uint32_t *pd;

  lock_acquire (&cur->parent_lock);
  if (cur->parent != NULL)
    {
      lock_acquire (&cur->parent->child_lock);
      struct list_elem* e = list_find_if (
          &cur->parent->child_list, process_child_pred, cur);
      lock_release (&cur->parent->child_lock);

      ASSERT (e != NULL);
      struct child_elem* child = list_entry (e, struct child_elem, elem);
      
      child->exit_status = cur->exit_status;
      child->child = NULL;
      sema_up (&child->sema);
    }
  lock_release (&cur->parent_lock);
  
  lock_acquire (&cur->child_lock);
  struct list_elem* e;
  while (!list_empty (&cur->child_list))
    {
      e = list_pop_front (&cur->child_list);
      struct child_elem* child = list_entry (e, struct child_elem, elem);
      if (child->child != NULL)
        {
          lock_acquire (&child->child->parent_lock);
          child->child->parent = NULL;
          lock_release (&child->child->parent_lock);
        }
      free (child);
    }
  lock_release (&cur->child_lock);

  /* Destroy the current process's page directory and switch back
     to the kernel-only page directory. */
  pd = cur->pagedir;
  if (pd != NULL) 
    {
      printf ("%s: exit(%d)\n", cur->name, cur->exit_status);    

#ifdef VM
      struct list_elem* e;
      while (!list_empty (&cur->mmap_list))
        {
          e = list_front (&cur->mmap_list);
          struct mmap_file* me = list_entry (e, struct mmap_file, elem);
          process_remove_mmap (me->mapid);
        }

      sup_page_table_destroy (&cur->sup_page_table);
#endif      

      /* Correct ordering here is crucial.  We must set
         cur->pagedir to NULL before switching page directories,
         so that a timer interrupt can't switch back to the
         process page directory.  We must activate the base page
         directory before destroying the process's page
         directory, or our active page directory will be one
         that's been freed (and cleared). */
      cur->pagedir = NULL;
      pagedir_activate (NULL);
      pagedir_destroy (pd);    
    }

  // Close all open files
  while (!list_empty (&cur->file_list))
    {
      e = list_pop_front (&cur->file_list);
      struct file_elem* file_elem = list_entry (e, struct file_elem, elem);
      lock_acquire (&fs_lock);
      file_close (file_elem->file);
      lock_release (&fs_lock);
      free (file_elem);
    }

  // Close the executable file
  if (cur->exec_file != NULL)
    {
      lock_acquire (&fs_lock);
      file_allow_write (cur->exec_file);
      file_close (cur->exec_file);
      lock_release (&fs_lock);
    }    
}

/* Sets up the CPU for running user code in the current
   thread.
   This function is called on every context switch. */
void
process_activate (void)
{
  struct thread *t = thread_current ();

  /* Activate thread's page tables. */
  pagedir_activate (t->pagedir);

  /* Set thread's kernel stack for use in processing
     interrupts. */
  tss_update ();
} 

static bool process_file_pred (const struct list_elem *e, void *aux);

int 
process_add_file (struct file *f)
{
  struct thread* cur = thread_current ();
  struct file_elem* fe = malloc (sizeof (struct file_elem));
  if (fe == NULL)
    return -1;
  int fd = (int) inode_get_inumber(file_get_inode(f));
  while(process_get_file(fd) != NULL)
    fd += FD_GROW_MAGIC;
  fe->file = f;
  fe->fd = fd;
  list_push_back (&cur->file_list, &fe->elem);
  return fe->fd;
}

static bool 
process_file_pred (const struct list_elem *e, void *aux)
{
  struct file_elem* fe = list_entry (e, struct file_elem, elem);
  return fe->fd == *(int*)aux;
}

struct file*
process_get_file (int fd)
{
  struct list_elem* e = list_find_if (&thread_current ()->file_list, 
      process_file_pred, &fd);
  if (e == NULL)
    return NULL;

  struct file_elem* fe = list_entry (e, struct file_elem, elem);
  return fe->file;
}

void
process_close_file (int fd)
{
  struct list_elem* e = list_find_if (&thread_current ()->file_list, 
      process_file_pred, &fd);
  if (e == NULL)
    return;
    
  struct file_elem* fe = list_entry (e, struct file_elem, elem);

  lock_acquire (&fs_lock);
  file_close (fe->file);
  lock_release (&fs_lock);
  list_remove (&fe->elem);
  free (fe);
}

static bool load_segment (
#ifdef VM
    enum page_location location, 
#endif
    struct file *file, off_t ofs, uint8_t *upage, uint32_t read_bytes, 
    uint32_t zero_bytes, bool writable);

#ifdef VM
int
process_add_mmap (struct file *f, void *addr)
{
  off_t size = file_length(f);
  if (size == 0) return MAPID_ERROR;

  if (page_overlaps(&thread_current()->sup_page_table, addr, size)) 
    return MAPID_ERROR;
  
  struct mmap_file *me = malloc(sizeof(struct mmap_file));
  if (me == NULL) return MAPID_ERROR;

  size_t read_bytes = size;
  size_t zero_bytes = (PGSIZE - (size % PGSIZE)) % PGSIZE;
  if (!load_segment(PAGE_LOC_FILESYS, f, 0, addr, 
      read_bytes, zero_bytes, true)) 
    {
      free(me);
      return MAPID_ERROR;
    }

  me->file = f;
  me->mapid = thread_current()->next_mapid;
  me->user_addr = addr;
  me->num_pages = (read_bytes + zero_bytes) / PGSIZE;
  thread_current()->next_mapid++;
  list_push_back(&thread_current()->mmap_list, &me->elem);
  return me->mapid;
}

static bool
process_mmap_pred (const struct list_elem *e, void *aux)
{
  struct mmap_file *me = list_entry(e, struct mmap_file, elem);
  return me->mapid == *(int*)aux;
}

struct mmap_file *
process_get_mmap (int mapid)
{
  struct list_elem *e = list_find_if(&thread_current()->mmap_list, 
      process_mmap_pred, &mapid);
  if (e == NULL) return NULL;

  struct mmap_file *me = list_entry(e, struct mmap_file, elem);
  return me;
}

void
process_remove_mmap (int mapid)
{
  struct mmap_file *me = process_get_mmap(mapid);
  if (me == NULL) return;

  for (size_t i = 0; i < me->num_pages; ++i)
    {
      struct sup_page_table_entry *entry = page_find(
          &thread_current()->sup_page_table, me->user_addr + i * PGSIZE);
      if (entry == NULL) continue;
      page_destroy(&thread_current()->sup_page_table, entry);
    }
  
  list_remove(&me->elem);
  free(me);
}
#endif

/* We load ELF binaries.  The following definitions are taken
   from the ELF specification, [ELF1], more-or-less verbatim.  */

/* ELF types.  See [ELF1] 1-2. */
typedef uint32_t Elf32_Word, Elf32_Addr, Elf32_Off;
typedef uint16_t Elf32_Half;

/* For use with ELF types in printf(). */
#define PE32Wx PRIx32   /* Print Elf32_Word in hexadecimal. */
#define PE32Ax PRIx32   /* Print Elf32_Addr in hexadecimal. */
#define PE32Ox PRIx32   /* Print Elf32_Off in hexadecimal. */
#define PE32Hx PRIx16   /* Print Elf32_Half in hexadecimal. */

/* Executable header.  See [ELF1] 1-4 to 1-8.
   This appears at the very beginning of an ELF binary. */
struct Elf32_Ehdr
  {
    unsigned char e_ident[16];
    Elf32_Half    e_type;
    Elf32_Half    e_machine;
    Elf32_Word    e_version;
    Elf32_Addr    e_entry;
    Elf32_Off     e_phoff;
    Elf32_Off     e_shoff;
    Elf32_Word    e_flags;
    Elf32_Half    e_ehsize;
    Elf32_Half    e_phentsize;
    Elf32_Half    e_phnum;
    Elf32_Half    e_shentsize;
    Elf32_Half    e_shnum;
    Elf32_Half    e_shstrndx;
  };

/* Program header.  See [ELF1] 2-2 to 2-4.
   There are e_phnum of these, starting at file offset e_phoff
   (see [ELF1] 1-6). */
struct Elf32_Phdr
  {
    Elf32_Word p_type;
    Elf32_Off  p_offset;
    Elf32_Addr p_vaddr;
    Elf32_Addr p_paddr;
    Elf32_Word p_filesz;
    Elf32_Word p_memsz;
    Elf32_Word p_flags;
    Elf32_Word p_align;
  };

/* Values for p_type.  See [ELF1] 2-3. */
#define PT_NULL    0            /* Ignore. */
#define PT_LOAD    1            /* Loadable segment. */
#define PT_DYNAMIC 2            /* Dynamic linking info. */
#define PT_INTERP  3            /* Name of dynamic loader. */
#define PT_NOTE    4            /* Auxiliary info. */
#define PT_SHLIB   5            /* Reserved. */
#define PT_PHDR    6            /* Program header table. */
#define PT_STACK   0x6474e551   /* Stack segment. */

/* Flags for p_flags.  See [ELF3] 2-3 and 2-4. */
#define PF_X 1          /* Executable. */
#define PF_W 2          /* Writable. */
#define PF_R 4          /* Readable. */

static bool setup_stack (void **esp, struct list* arg_list);
static bool validate_segment (const struct Elf32_Phdr *, struct file *);

static bool setup_args (void **esp, struct list* arg_list);                        

/* Loads an ELF executable from FILE_NAME into the current thread.
   Stores the executable's entry point into *EIP
   and its initial stack pointer into *ESP.
   Returns true if successful, false otherwise. */
bool
load (struct list* arg_list, void (**eip) (void), void **esp) 
{
  struct thread *t = thread_current ();
  struct Elf32_Ehdr ehdr;
  struct file *file = NULL;
  off_t file_ofs;
  bool success = false;
  int i;

  /* Allocate and activate page directory. */
  t->pagedir = pagedir_create ();
  if (t->pagedir == NULL) 
    goto done;
  process_activate ();

  // Get the program name from the arg_list
  char* exec_name = list_entry(
      list_front(arg_list), struct arg_elem, elem)->arg;

  /* Open executable file. */
  lock_acquire(&fs_lock);
  file = filesys_open (exec_name, ROOT_DIR_FD);
  if (file == NULL) 
    {
      printf ("load: %s: open failed\n", exec_name);
      goto done; 
    }

  /* Read and verify executable header. */
  if (file_read (file, &ehdr, sizeof ehdr) != sizeof ehdr
      || memcmp (ehdr.e_ident, "\177ELF\1\1\1", 7)
      || ehdr.e_type != 2
      || ehdr.e_machine != 3
      || ehdr.e_version != 1
      || ehdr.e_phentsize != sizeof (struct Elf32_Phdr)
      || ehdr.e_phnum > 1024) 
    {
      printf ("load: %s: error loading executable\n", exec_name);
      goto done; 
    }

  /* Read program headers. */
  file_ofs = (off_t)ehdr.e_phoff;
  for (i = 0; i < ehdr.e_phnum; i++) 
    {
      struct Elf32_Phdr phdr;

      if (file_ofs < 0 || file_ofs > file_length (file))
        goto done;
      file_seek (file, file_ofs);

      if (file_read (file, &phdr, sizeof phdr) != sizeof phdr)
        goto done;
      file_ofs += sizeof phdr;
      switch (phdr.p_type) 
        {
        case PT_NULL:
        case PT_NOTE:
        case PT_PHDR:
        case PT_STACK:
        default:
          /* Ignore this segment. */
          break;
        case PT_DYNAMIC:
        case PT_INTERP:
        case PT_SHLIB:
          goto done;
        case PT_LOAD:
          if (validate_segment (&phdr, file)) 
            {
              bool writable = (phdr.p_flags & PF_W) != 0;
              uint32_t file_page = phdr.p_offset & ~PGMASK;
              uint32_t mem_page = phdr.p_vaddr & ~PGMASK;
              uint32_t page_offset = phdr.p_vaddr & PGMASK;
              uint32_t read_bytes, zero_bytes;
              if (phdr.p_filesz > 0)
                {
                  /* Normal segment.
                     Read initial part from disk and zero the rest. */
                  read_bytes = page_offset + phdr.p_filesz;
                  zero_bytes = (ROUND_UP (page_offset + phdr.p_memsz, PGSIZE)
                                - read_bytes);
                }
              else 
                {
                  /* Entirely zero.
                     Don't read anything from disk. */
                  read_bytes = 0;
                  zero_bytes = ROUND_UP (page_offset + phdr.p_memsz, PGSIZE);
                }
              if (!load_segment (
#ifdef VM                
                    PAGE_LOC_EXEC, 
#endif                    
                    file, (off_t)file_page, (void *) mem_page,
                    read_bytes, zero_bytes, writable))
                goto done;
            }
          else
            goto done;
          break;
        }
    }

  /* Set up stack. */
  if (!setup_stack (esp, arg_list))
    goto done;

  /* Start address. */
  *eip = (void (*) (void)) ehdr.e_entry;

  // Deny write to the executable file
  file_deny_write(file);
  t->exec_file = file;

  success = true;

 done:
  /* We arrive here whether the load is successful or not. */
  if (!success) 
    file_close (file);
  lock_release(&fs_lock);
  return success;
}

/* load() helpers. */

/* Checks whether PHDR describes a valid, loadable segment in
   FILE and returns true if so, false otherwise. */
static bool
validate_segment (const struct Elf32_Phdr *phdr, struct file *file) 
{
  /* p_offset and p_vaddr must have the same page offset. */
  if ((phdr->p_offset & PGMASK) != (phdr->p_vaddr & PGMASK)) 
    return false; 

  /* p_offset must point within FILE. */
  if (phdr->p_offset > (Elf32_Off) file_length (file)) 
    return false;

  /* p_memsz must be at least as big as p_filesz. */
  if (phdr->p_memsz < phdr->p_filesz) 
    return false; 

  /* The segment must not be empty. */
  if (phdr->p_memsz == 0)
    return false;
  
  /* The virtual memory region must both start and end within the
     user address space range. */
  if (!is_user_vaddr ((void *) phdr->p_vaddr))
    return false;
  if (!is_user_vaddr ((void *) (phdr->p_vaddr + phdr->p_memsz)))
    return false;

  /* The region cannot "wrap around" across the kernel virtual
     address space. */
  if (phdr->p_vaddr + phdr->p_memsz < phdr->p_vaddr)
    return false;

  /* Disallow mapping page 0.
     Not only is it a bad idea to map page 0, but if we allowed
     it then user code that passed a null pointer to system calls
     could quite likely panic the kernel by way of null pointer
     assertions in memcpy(), etc. */
  if (phdr->p_vaddr < PGSIZE)
    return false;

  /* It's okay. */
  return true;
}

/* Loads a segment starting at offset OFS in FILE at address
   UPAGE.  In total, READ_BYTES + ZERO_BYTES bytes of virtual
   memory are initialized, as follows:

        - READ_BYTES bytes at UPAGE must be read from FILE
          starting at offset OFS.

        - ZERO_BYTES bytes at UPAGE + READ_BYTES must be zeroed.

   The pages initialized by this function must be writable by the
   user process if WRITABLE is true, read-only otherwise.

   Return true if successful, false if a memory allocation error
   or disk read error occurs. */
static bool
load_segment (
#ifdef VM
    enum page_location location, 
#endif
    struct file *file, off_t ofs, uint8_t *upage, 
    uint32_t read_bytes, uint32_t zero_bytes, bool writable) 
{
  ASSERT ((read_bytes + zero_bytes) % PGSIZE == 0);
  ASSERT (pg_ofs (upage) == 0);
  ASSERT (ofs % PGSIZE == 0);

#ifdef VM
  ASSERT (location == PAGE_LOC_EXEC || location == PAGE_LOC_FILESYS);
#endif

#ifndef VM
  file_seek (file, ofs);
#endif
  while (read_bytes > 0 || zero_bytes > 0) 
    {
      /* Calculate how to fill this page.
         We will read PAGE_READ_BYTES bytes from FILE
         and zero the final PAGE_ZERO_BYTES bytes. */
      size_t page_read_bytes = read_bytes < PGSIZE ? read_bytes : PGSIZE;
      size_t page_zero_bytes = PGSIZE - page_read_bytes;

#ifdef VM
      struct sup_page_table_entry* spte = NULL;
      if (page_read_bytes != 0)
        spte = page_create (&thread_current ()->sup_page_table, upage, 
          location, NULL, BITMAP_ERROR, file, ofs, 
          page_read_bytes, page_zero_bytes, writable);
      else
        spte = page_create (&thread_current ()->sup_page_table, upage, 
          PAGE_LOC_ZERO, NULL, BITMAP_ERROR, NULL, 0, 0, 0, writable);

      if (spte == NULL) return false;
      ofs += (off_t)page_read_bytes;
#else
      /* Get a page of memory. */
      uint8_t *kpage = palloc_get_page (PAL_USER);
      if (kpage == NULL)
        return false;

      /* Load this page. */
      if (file_read (file, kpage, (off_t)page_read_bytes) != (int) page_read_bytes)
        {
          palloc_free_page (kpage);
          return false; 
        }
      memset (kpage + page_read_bytes, 0, page_zero_bytes);

      /* Add the page to the process's address space. */
      if (!install_page (upage, kpage, writable)) 
        {
          palloc_free_page (kpage);
          return false; 
        }
#endif

      /* Advance. */
      read_bytes -= page_read_bytes;
      zero_bytes -= page_zero_bytes;
      upage += PGSIZE;
    }
  return true;
}

/* Create a minimal stack by mapping a zeroed page at the top of
   user virtual memory. */
static bool
setup_stack (void **esp, struct list* arg_list) 
{
  bool success = false;

#ifdef VM
  struct sup_page_table_entry* spte = page_alloc (
      &thread_current ()->sup_page_table, ((uint8_t *) PHYS_BASE) - PGSIZE, true);
  if (spte != NULL)
    {
      success = setup_args (esp, arg_list); 

      if (!success)
        page_destroy (&thread_current ()->sup_page_table, spte);
    }
#else
  uint8_t *kpage = palloc_get_page (PAL_USER | PAL_ZERO);
  if (kpage != NULL) 
    {
      success = install_page (((uint8_t *) PHYS_BASE) - PGSIZE, kpage, true);
      if (success)
        success = setup_args (esp, arg_list); 

      if (!success)
        palloc_free_page (kpage);
    }
#endif

  return success;
}

static bool 
setup_args (void **esp, struct list* arg_list) {
  if (list_size (arg_list) > 32)
    return true;

  *esp = PHYS_BASE;
  for (struct list_elem* e = list_rbegin (arg_list); 
        e != list_rend (arg_list); e = list_prev (e)) {
    struct arg_elem* arg = list_entry (e, struct arg_elem, elem);
    size_t len = strlen (arg->arg);
    *esp -= len + 1;
    memcpy (*esp, arg->arg, len + 1);
    arg->addr = *esp;
  }

  int word_align = (size_t)(*esp) % 4;
  *esp -= word_align;
  memset (*esp, 0, word_align);

  *esp -= sizeof (char*);
  memset (*esp, 0, sizeof (char*));

  for (struct list_elem* e = list_rbegin (arg_list); 
        e != list_rend (arg_list); e = list_prev (e)) {
    struct arg_elem* arg = list_entry (e, struct arg_elem, elem);
    *esp -= sizeof (char*);
    memcpy (*esp, &arg->addr, sizeof (char*));
  }

  char* argv = *esp;
  *esp -= sizeof (char**);
  memcpy (*esp, &argv, sizeof (char**));

  *esp -= sizeof (int);
  int argc = (int)list_size (arg_list);
  memcpy (*esp, &argc, sizeof (int));

  *esp -= sizeof (void*);
  memset (*esp, 0, sizeof (void*));

  return true;
}

/* Adds a mapping from user virtual address UPAGE to kernel
   virtual address KPAGE to the page table.
   If WRITABLE is true, the user process may modify the page;
   otherwise, it is read-only.
   UPAGE must not already be mapped.
   KPAGE should probably be a page obtained from the user pool
   with palloc_get_page().
   Returns true on success, false if UPAGE is already mapped or
   if memory allocation fails. */
bool
install_page (void *upage, void *kpage, bool writable)
{
  struct thread *t = thread_current ();

  /* Verify that there's not already a page at that virtual
     address, then map our page there. */
  return (pagedir_get_page (t->pagedir, upage) == NULL
          && pagedir_set_page (t->pagedir, upage, kpage, writable));
}
