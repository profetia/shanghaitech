#ifndef USERPROG_PROCESS_H
#define USERPROG_PROCESS_H

#include "threads/synch.h"
#include "threads/thread.h"

extern struct lock fs_lock;

void process_init (void);
tid_t process_execute (const char *file_name,int cwd_fd);
int process_wait (tid_t);
void process_exit (void);
void process_activate (void);

bool install_page (void *upage, void *kpage, bool writable);

int process_add_file (struct file *f);
struct file *process_get_file (int fd);
void process_close_file (int fd);

#ifdef VM
struct mmap_file
  {
    int mapid;
    void* user_addr;
    struct file *file;
    size_t num_pages;
    struct list_elem elem;
  };

int process_add_mmap (struct file *f, void *addr);
struct mmap_file *process_get_mmap (int mapid);
void process_remove_mmap (int mapid);
#endif

#endif /* userprog/process.h */
