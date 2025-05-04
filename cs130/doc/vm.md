            +---+
            |          CS 140          |
            | PROJECT 3: VIRTUAL MEMORY |
            |      DESIGN DOCUMENT      |
            +---+

#### GROUP 

> Fill in the names and email addresses of your group members.

Linshu Yang <yanglsh@shanghaitech.edu.cn>

Lei Huang <huanglei@shanghaitech.edu.cn>

#### PRELIMINARIES 

> If you have any preliminary comments on your submission, notes for the
> TAs, or extra credit, please give them here.

> Please cite any offline or online sources you consulted while
> preparing your submission, other than the Pintos documentation, course
> text, lecture notes, and course staff.

            PAGE TABLE MANAGEMENT
            =====================

#### DATA STRUCTURES 

> A1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c 
// threads/thread.h

...

struct thread
  {
    ...
    struct hash sup_page_table;         /* Supplemental page table. */
    ...
  }

...

// vm/page.h

...

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

  bool writable;

  bool dirty;
  bool accessed;  

  struct lock* lock;
  struct hash_elem elem;
};

...

// vm/frame.h

...

struct frame_table_entry {
    uint32_t *frame;
    struct thread *owner;
    struct sup_page_table_entry *page_entry;

    struct list_elem elem;
};

...

// vm/frame.c

...

static struct list frame_table;
static struct lock frame_table_lock;

...

```

- In `struct thread`:
    - `sup_page_table`: A hash table that stores the supplemental page table of the thread.

- `enum page_location`: An enum that represents the location of a page.
    - `PAGE_LOC_ZERO`: Page is all zeros, no other fields are valid.
    - `PAGE_LOC_SWAP`: Page is in swap, swap_index is valid.
    - `PAGE_LOC_MEMORY`: Page is in memory, frame_entry is valid.
    - `PAGE_LOC_EXEC`: Page is in the executable, file and file_offset are valid. Executable pages cannot be written back to the file system instead they must be evicted to swap.
    - `PAGE_LOC_FILESYS`: Page is in the file system, file and file_offset are valid.
    - `PAGE_LOC_MMAPPED`: Page is in a memory mapped file, frame_entry and file are valid.
    - `PAGE_LOC_ERROR`: Error.

- `struct sup_page_table_entry`: A struct that represents an entry in the supplemental page table.
    - `user_vaddr`: The user virtual address of the page.
    - `location`: The location of the page.
    - `frame_entry`: The frame table entry of the page.
    - `writable`: Whether the page is writable.
    - `dirty`: Whether the page is dirty.
    - `accessed`: Whether the page is accessed.
    - `lock`: The lock of the page.
    - `elem`: The hash element of the page.

- `struct frame_table_entry`: A struct that represents an entry in the frame table.
    - `frame`: The frame of the page.
    - `owner`: The owner of the page.
    - `page_entry`: The supplemental page table entry of the page.
    - `elem`: The list element of the page.

- `struct list frame_table`: A list that stores the frame table.
- `struct lock frame_table_lock`: The lock of the frame table.

#### ALGORITHMS 

> A2: In a few paragraphs, describe your code for accessing the data
> stored in the SPT about a given page.

First, the SPT entry is found given the user virtual address of the page. Then, the location of the page is checked. If the page is in memory or mapped into memory, the corresponding memory is accessed directly. If the page is in swap, a new frame of memory is allocated and the page is read from the swap to the memory. If the page is in the executable, the page is read from the executable to the memory. If the page is in the file system, the page is read from the file system to the memory. 

> A3: How does your code coordinate accessed and dirty bits between
> kernel and user virtual addresses that alias a single frame, or
> alternatively how do you avoid the issue?

Our code coordinates accessed and dirty bits between kernel and user virtual addresses by adding a dirty bit and an accessed bit to the SPT entry. When handling a page fault, the dirty bit and the accessed bit are set accordingly. When a page is evicted, the dirty bit in SPT along with the dirty bit in PTE are checked. If either of them is set, the page is written back to the file system or swap.

#### SYNCHRONIZATION 

> A4: When two user processes both need a new frame at the same time,
> how are races avoided?

When two user processes both need a new frame at the same time, the frame table lock is acquired. Then, the frame table is checked for a free frame. If there is a free frame, the frame is allocated to the process. If there is no free frame, a frame is evicted from the frame table. The frame table lock is released after the frame is allocated or evicted.

#### RATIONALE 

> A5: Why did you choose the data structure(s) that you did for
> representing virtual-to-physical mappings?

We chose a hash table to represent the virtual-to-physical mappings because it is easy to find the corresponding entry given the user virtual address of the page. We chose a list to represent the frame table because it is easy to find a free frame in the frame table.

               PAGING TO AND FROM DISK
               =======================

#### DATA STRUCTURES 

> B1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// vm/page.h

...

struct sup_page_table_entry {
  ...
  size_t swap_index;
  ...
};

...

// vm/swap.h

...

static struct block *swap_block;
static struct bitmap *swap_bitmap;
static struct lock swap_lock;

...

```

- In `struct sup_page_table_entry`:
    - `swap_index`: The swap index of the page.

- `struct block *swap_block`: The swap block.
- `struct bitmap *swap_bitmap`: The swap bitmap.
- `struct lock swap_lock`: The lock of the swap.

#### ALGORITHMS 

> B2: When a frame is required but none is free, some frame must be
> evicted.  Describe your code for choosing a frame to evict.

The frame table is iterated through to find a frame to evict. The frame is chosen by the clock algorithm. If the frame is accessed, the accessed bit is cleared. Otherwise, the frame is evicted. When no frame is found, the evict victim fallback to the front of the frame table.

> B3: When a process P obtains a frame that was previously used by a
> process Q, how do you adjust the page table (and any other data
> structures) to reflect the frame Q no longer has?

When process Q no longer possesses a frame, the frame is freed with `frame_free()`. The `location` field in the corresponding SPT entry is set to Q's new location and the `frame_entry` field is set to NULL.

> B4: Explain your heuristic for deciding whether a page fault for an
> invalid virtual address should cause the stack to be extended into
> the page that faulted.

If the fault address is below PHYS_BASE and above the `STACK_BOTTOM` which is 0x08048000, and it is above `esp - 32`, the stack is extended into the page that faulted.

#### SYNCHRONIZATION 

> B5: Explain the basics of your VM synchronization design.  In
> particular, explain how it prevents deadlock.  (Refer to the
> textbook for an explanation of the necessary conditions for
> deadlock.)

The VM synchronization design uses a lock for each SPT entry. Whenever any operation is performed on a SPT entry, the corresponding lock is acquired. In these operations, the process may at most wait for the lock of the global frame table lock or the lock of the swap, which does not wait for any other locks. Since there is no `wait while holding` in the global frame table lock or the lock of the swap, there is no possibility of deadlock.

> B6: A page fault in process P can cause another process Q's frame
> to be evicted.  How do you ensure that Q cannot access or modify
> the page during the eviction process?  How do you avoid a race
> between P evicting Q's frame and Q faulting the page back in?

When a page fault in process P causes another process Q's frame to be evicted, the lock in the corresponding SPT entry is acquired. Q's frame cannot be evicted until the lock is released, signaling the completion of Q's page fault handling. When Q faults the page back in, the lock is acquired again. Q cannot access or modify the page during the eviction process because the lock is held by P.

> B7: Suppose a page fault in process P causes a page to be read from
> the file system or swap.  How do you ensure that a second process Q
> cannot interfere by e.g. attempting to evict the frame while it is
> still being read in?

Similar to B6, when a page fault in process P causes a page to be read from the file system or swap, the lock in the corresponding SPT entry is acquired. Q cannot interfere by attempting to evict the frame while it is still being read in because the lock is held by P.

> B8: Explain how you handle access to paged-out pages that occur
> during system calls.  Do you use page faults to bring in pages (as
> in user programs), or do you have a mechanism for "locking" frames
> into physical memory, or do you use some other design?  How do you
> gracefully handle attempted accesses to invalid virtual addresses?

When a page fault occurs during system calls, the page is brought in by the page fault handler. We do not use a mechanism for "locking" frames into physical memory. If the virtual address is not in the SPT, the process is terminated. 

#### RATIONALE 

> B9: A single lock for the whole VM system would make
> synchronization easy, but limit parallelism.  On the other hand,
> using many locks complicates synchronization and raises the
> possibility for deadlock but allows for high parallelism.  Explain
> where your design falls along this continuum and why you chose to
> design it this way.

Our design falls along the continuum of using many locks. We chose to design it this way because of our preference for high parallelism. We believe that the possibility of deadlock is low and can be avoided by careful design.

             MEMORY MAPPED FILES
             ===================

#### DATA STRUCTURES 

> C1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// threads/thread.h

...

struct thread
  {
    ...
    struct list mmap_list;              /* List of mmap files. */
    int next_mapid;                     /* Next mapid. */
    ...
  }

...

// vm/page.h

...

struct sup_page_table_entry {
  ...
  struct file* file;
  off_t file_offset;
  size_t read_bytes;
  size_t zero_bytes;
  ...
};

...

```

- In `struct thread`:
    - `mmap_list`: A list that stores the memory mapped files of the thread.
    - `next_mapid`: The next mapid of the thread.

- In `struct sup_page_table_entry`:
    - `file`: The file of the page.
    - `file_offset`: The file offset of the page.
    - `read_bytes`: The number of bytes to read from the file.
    - `zero_bytes`: The number of bytes to zero after the file.

#### ALGORITHMS 

> C2: Describe how memory mapped files integrate into your virtual
> memory subsystem.  Explain how the page fault and eviction
> processes differ between swap pages and other pages.

Memory mapped files integrate into our virtual memory subsystem by adding a new location `PAGE_LOC_MMAPPED` to the `enum page_location`. When a page fault occurs, the page is brought in by the page fault handler. When a page is evicted, the page is written back to the file system if it is dirty. The eviction process differs between swap pages and other pages in that swap pages are written back to the swap instead of the file system.

> C3: Explain how you determine whether a new file mapping overlaps
> any existing segment.

In our implementation, we do not allow a new file mapping to overlap any existing segment. When creating a new file mapping, the file is checked for overlapping by iterating through the virtual address space of the process by `PGSIZE` bytes. If any part of the file overlaps with an existing segment, the file mapping is not created.

#### RATIONALE 

> C4: Mappings created with "mmap" have similar semantics to those of
> data demand-paged from executables, except that "mmap" mappings are
> written back to their original files, not to swap.  This implies
> that much of their implementation can be shared.  Explain why your
> implementation either does or does not share much of the code for
> the two situations.

Our implementation shares much of the code for the two situations. The page fault handler for memory mapped files is similar to that for executables. The eviction process for memory mapped files is similar to that for executables. The only difference is that the location of the page is set to `PAGE_LOC_MMAPPED` instead of `PAGE_LOC_MEMORY`.

               SURVEY QUESTIONS
               ================

Answering these questions is optional, but it will help us improve the
course in future quarters.  Feel free to tell us anything you
want--these questions are just to spur your thoughts.  You may also
choose to respond anonymously in the course evaluations at the end of
the quarter.

> In your opinion, was this assignment, or any one of the three problems
> in it, too easy or too hard?  Did it take too long or too little time?

> Did you find that working on a particular part of the assignment gave
> you greater insight into some aspect of OS design?

> Is there some particular fact or hint we should give students in
> future quarters to help them solve the problems?  Conversely, did you
> find any of our guidance to be misleading?

> Do you have any suggestions for the TAs to more effectively assist
> students, either for future quarters or the remaining projects?

> Any other comments?
