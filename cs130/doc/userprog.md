             +--------------------------+
             |          CS 140          |
             | PROJECT 2: USER PROGRAMS |
             |     DESIGN DOCUMENT      |
             +--------------------------+

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

               ARGUMENT PASSING
               ================

#### DATA STRUCTURES 

> A1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// userprog/process.c

...

struct arg_elem 
  {
    char* arg;
    char* addr;
    struct list_elem elem;
  };

...

```

- `struct arg_elem`: A struct to store the argument and its address in the stack.
    - `arg`: The argument.
    - `addr`: The address of the argument in the stack.
    - `elem`: The list element.

#### ALGORITHMS 

> A2: Briefly describe how you implemented argument parsing. How do
> you arrange for the elements of argv[] to be in the right order?
> How do you avoid overflowing the stack page?

Arguments are parsed using `strtok_r()`, which is stored in a list. This list is then reversed and stored in the stack, guaranteeing the right order. Stack overflow is avoided by restricting the number of arguments.

#### RATIONALE 

> A3: Why does Pintos implement strtok_r() but not strtok()?

`strtok_r()` is thread-safe, while `strtok()` is not.

> A4: In Pintos, the kernel separates commands into a executable name
> and arguments.  In Unix-like systems, the shell does this
> separation.  Identify at least two advantages of the Unix approach.

- The shell can do more complex parsing, such as parsing `|` and `>`.
- The shell can do more complex redirection, such as redirecting `stderr` to a file.

                 SYSTEM CALLS
                 ============

#### DATA STRUCTURES 

> B1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// threads/thread.h

...

struct thread
  {
    ...
    int exit_status;                    /* Exit status. */
    struct file *exec_file;             /* Executable file. */
    struct list file_list;              /* List of files. */
    int next_fd;                        /* Next file descriptor. */

    struct list child_list;             /* List of children. */    
    struct lock child_lock;             /* Child lock. */
    struct thread *parent;              /* Parent thread. */
    struct lock parent_lock;            /* Parent lock. */
    ...    
  }

...

// userprog/process.h

...

struct lock fs_lock;

...

// userprog/process.c

...

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

...

```

- In `struct thread`
  - `exit_status`: The exit status of the thread.
  - `exec_file`: The executable file of the thread.
  - `file_list`: The list of files opened by the thread.
  - `next_fd`: The next file descriptor to be assigned.
  - `child_list`: The list of children of the thread.
  - `child_lock`: The lock for the child list.
  - `parent`: The parent of the thread.
  - `parent_lock`: The lock for the parent.

-  `fs_lock`: The lock for the file system.

- `struct child_elem`: A struct to store the child and its exit status.
  - `child`: The child thread.
  - `exit_status`: The exit status of the child.
  - `sema`: The semaphore for the child.
  - `pid`: The pid of the child.
  - `elem`: The list element.

- `struct start_process_args`: A struct to store the arguments for `start_process()`.
  - `arg_list`: The list of arguments.
  - `parent`: The parent thread.
  - `child`: The child element.

> B2: Describe how file descriptors are associated with open files.
> Are file descriptors unique within the entire OS or just within a
> single process?

Open files are stored in a list in the thread struct. When a file is opened, a file descriptor is assigned to it. File descriptors are unique within a single process.

#### ALGORITHMS 

> B3: Describe your code for reading and writing user data from the
> kernel.

Accessing user data from the kernel is done directly since interrupts does not change the page directory. Before accessing user data, the kernel checks whether the user pointer is valid using `pagedir_get_page()`. When the user pointer is invalid, the process is terminated.

> B4: Suppose a system call causes a full page (4,096 bytes) of data
> to be copied from user space into the kernel.  What is the least
> and the greatest possible number of inspections of the page table
> (e.g. calls to pagedir_get_page()) that might result?  What about
> for a system call that only copies 2 bytes of data?  Is there room
> for improvement in these numbers, and how much?

For a system call that copies 4096 bytes of data, the  greatest number of inspections is 4096, since every byte of the page is inspected. The
least number of inspections is 2, as the bytes can at most be on 2 pages, which can be inspected by looking up the pages twice. For a system call that copies 2 bytes of data, the least number of inspections is 1 and the greatest number of inspections is 2. 

> B5: Briefly describe your implementation of the "wait" system call
> and how it interacts with process termination.

The `wait()` system call is implemented by using a semaphore. When a child process is created, a semaphore is initialized to 0. When the child process exits, the semaphore is upped. When the parent process calls `wait()`, it waits for the semaphore to be upped. When the child process is terminated, the semaphore is upped. If the child process is terminated before the parent process calls `wait()`, its exit status is stored in the child element and used directly when the parent process calls `wait()`.

> B6: Any access to user program memory at a user-specified address
> can fail due to a bad pointer value.  Such accesses must cause the
> process to be terminated.  System calls are fraught with such
> accesses, e.g. a "write" system call requires reading the system
> call number from the user stack, then each of the call's three
> arguments, then an arbitrary amount of user memory, and any of
> these can fail at any point.  This poses a design and
> error-handling problem: how do you best avoid obscuring the primary
> function of code in a morass of error-handling?  Furthermore, when
> an error is detected, how do you ensure that all temporarily
> allocated resources (locks, buffers, etc.) are freed?  In a few
> paragraphs, describe the strategy or strategies you adopted for
> managing these issues.  Give an example.

Every byte of user memory is validated before it is accessed using `pagedir_get_page()`. When a bad pointer value is detected, the process is terminated. Validation is done before any other operations are done, so that no resources are allocated and need to be freed. Locks and buffers that are used is owned by the process, so they are freed when the process exits.

Example: When a system call to `read` is made, the system call code is validated and read from the user stack. Then the three arguments: the file descriptor, the buffer and the size is validated and read from the user stack. The buffer is then validated by its start and end address. The file descriptor is validated by checking if it is in the file list. 
Finally, the file is read from the file system and the number of bytes read is returned.

#### SYNCHRONIZATION 

> B7: The "exec" system call returns -1 if loading the new executable
> fails, so it cannot return before the new executable has completed
> loading.  How does your code ensure this?  How is the load
> success/failure status passed back to the thread that calls "exec"?

Before the new thread is created, the parent creates a `child_elem` and passes it to the child thread. After the child thread is created, the parent waits for the child to load the executable by waiting for the semaphore in the `child_elem` to be upped. When the child thread is done loading the executable, it upps the semaphore. If the executable fails to load, the child thread exits with status -1, setting the `pid` in the `child_elem` to `TID_ERROR`. When the parent thread is done waiting, it checks the `pid` in the `child_elem`. If the `pid` is `TID_ERROR`, the parent thread returns -1. Otherwise, the parent thread returns the `pid` of the child thread.

> B8: Consider parent process P with child process C.  How do you
> ensure proper synchronization and avoid race conditions when P
> calls wait(C) before C exits?  After C exits?  How do you ensure
> that all resources are freed in each case?  How about when P
> terminates without waiting, before C exits?  After C exits?  Are
> there any special cases?

The exit status of the child process is stored in the `child_elem`, which is stored in the `child_list` owned by the parent. Either P waits C before or after it exits causes no race conditions as it only reads the exit status after the semaphore in `child_elem` is upped. And the semaphore is upped when C exits. When P terminates without waiting, the `child_list` is freed, which frees the `child_elem` and the semaphore. If P terminates before C exits, the `parent` pointer in C is set to NULL. The special case is when P and C exits at almost the same time, then the `parent_lock` is used to ensure that either C's exit status is stored in the `child_elem` or the `parent` pointer in C is set to NULL.

#### RATIONALE 

> B9: Why did you choose to implement access to user memory from the
> kernel in the way that you did?

Because it is simple and easy to implement, as it is already implemented in `pagedir_get_page()`.

> B10: What advantages or disadvantages can you see to your design
> for file descriptors?

Advantages: It is simple and easy to implement. It is also easy to understand and debug.
Disadvantages: It is not very efficient, as it requires a linear search through the file list to find the file descriptor.

> B11: The default tid_t to pid_t mapping is the identity mapping.
> If you changed it, what advantages are there to your approach?

We did not change it.

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
