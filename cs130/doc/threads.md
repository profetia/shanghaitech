            +--------------------+
            |        CS 140      |
            | PROJECT 1: THREADS |
            |   DESIGN DOCUMENT  |
            +--------------------+

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

                 ALARM CLOCK
                 ===========

#### DATA STRUCTURES 

> A1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// src/thread/thread.h

...

struct thread
  {
    ...
    int64_t wakeup_tick;                /* Wakeup tick. */
    struct list_elem sleep_elem;        /* List element for sleep threads list. */
    ...
  };

...

/* List of sleeping threads. Processes are added to this list
   when they are sleeping and removed when they are woken up. */
static struct list sleep_list;

...

```

- In `struct thread`
    - `wakeup_tick` : The time tick when the thread should wakeup.
    - `sleep_elem` : the `list_elem` to be inserted into the `sleep_list`.
- `sleep_list`: the list to store the sleeping threads.

#### ALGORITHMS 

> A2: Briefly describe what happens in a call to timer_sleep(),
> including the effects of the timer interrupt handler.

In a call to `timer_sleep`, the function will calculate the wakeup tick then call `thread_sleep`. The function `thread_sleep` will first insert the thread into the `sleep_list` in an interrupt disabled context to avoid race, then call `thread_block` to block the thread. 

When the timer interrupt handler is called, it will iterate the `sleep_list` to find the threads that should be woken up, then call `thread_unblock` to unblock the threads. This is in an interrupt disabled context to avoid race.

> A3: What steps are taken to minimize the amount of time spent in
> the timer interrupt handler?

To minimize the amount of time spent in the timer interrupt handler, the `sleep_list` is maintained to be ordered in ascending order of `wakeup_tick`. So when the timer interrupt handler is called, it only needs to iterate the `sleep_list` from the head to find the threads that should be woken up and break the loop when the `wakeup_tick` of the thread is greater than the current tick. 

#### SYNCHRONIZATION 

> A4: How are race conditions avoided when multiple threads call
> timer_sleep() simultaneously?

They are avoided by performing the operations on the `sleep_list` in an interrupt disabled context.

> A5: How are race conditions avoided when a timer interrupt occurs
> during a call to timer_sleep()?

They are avoided by performing the operations on the `sleep_list` in an interrupt disabled context.

#### RATIONALE 

> A6: Why did you choose this design?  In what ways is it superior to
> another design you considered?

We choose this design because it is simple and efficient. Compared to the other lock-based design, it is simpler and more efficient because it does not need to acquire a lock when iterating the `sleep_list` in the timer interrupt handler.

                 PRIORITY SCHEDULING
                 ===================

#### DATA STRUCTURES 

> B1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// src/thread/thread.h

...

struct thread
  {
    ...
    int init_priority;                  /* Initial priority. */
    struct list donor_list;             /* List of donors. */
    struct list_elem donor_elem;        /* List element for donors list. */
    struct lock *waiting_lock;          /* Lock that the thread is waiting on. */  
    ...    
  }

...

```

- In `struct thread`
    - `init_priority` : The initial priority of the thread.
    - `donor_list` : The list of donors.
    - `donor_elem` : The `list_elem` to be inserted into the `donor_list`.
    - `waiting_lock` : The lock that the thread is waiting on.

> B2: Explain the data structure used to track priority donation.
> Use ASCII art to diagram a nested donation.  (Alternately, submit a
> .png file.)

Linked list is used to track priority donation. The `donor_list` in `struct thread` is used to store the donors. The `donor_elem` in `struct thread` is used to be inserted into the `donor_list`. The `waiting_lock` in `struct thread` is used to store the lock that the thread is waiting on.
With the above data structures, the priority donation can be tracked since the thread's donors and the lock its waiting on can be tracked.

#### ALGORITHMS 

> B3: How do you ensure that the highest priority thread waiting for
> a lock, semaphore, or condition variable wakes up first?

When waking up a thread, the thread with the highest priority will be woken up first, which is achieved by `list_max`. So the highest priority thread waiting for a lock, semaphore, or condition variable will wake up first.

> B4: Describe the sequence of events when a call to lock_acquire()
> causes a priority donation.  How is nested donation handled?

When a call to `lock_aquire` causes a priority donation, the thread will first donate its priority to the thread holding the lock, then the thread holding the lock will donate its priority to the thread holding the lock it is waiting on, and so on. This is achieved by `thread_donate_priority` and `thread_forward_priority`. Nested donation is handled by `thread_forward_priority`.

> B5: Describe the sequence of events when lock_release() is called
> on a lock that a higher-priority thread is waiting for.

When `lock_release` is called on a lock that a higher-priority thread is waiting for, the thread holding the lock will remove the threads waiting on the lock from its `donor_list` and restore its priority to its `init_priority`. 
Then the thread holding the lock will call `sema_up` to release the lock, which will wake up the thread with the highest priority waiting on the lock. This is achieved by `list_max`.

#### SYNCHRONIZATION 

> B6: Describe a potential race in thread_set_priority() and explain
> how your implementation avoids it.  Can you use a lock to avoid
> this race?

In `thread_set_priority`, `thread_pushup_priority` will be called to update the priority with the highest donation, iterating over the `donor_list`. Meanwhile, `donor_list` could be modified in this process, causing a possible race.

However, it is avoided as the `donor_list` can only be modified in `lock_aquire` and `lock_release`, where the operation on `donor_list` is done in an interrupt disabled context.

#### RATIONALE 

> B7: Why did you choose this design?  In what ways is it superior to
> another design you considered?

Because it is simple and efficient. Compared to the other design, it is simpler to only track the donors and the lock that the thread is waiting on, and transverse the chain of donations recursively to forward the priority upstream.

              ADVANCED SCHEDULER
              ==================

#### DATA STRUCTURES 

> C1: Copy here the declaration of each new or changed `struct' or
> `struct' member, global or static variable, `typedef', or
> enumeration.  Identify the purpose of each in 25 words or less.

```c
// src/threads/fixed-point.h

...

typedef int f32;

...

// src/threads/thread.h

...

struct thread
  {
    ...
    int nice;                           /* Nice value. */
    f32 recent_cpu;                     /* Recent CPU. */
    ...
  }

...

```

- `f32` : The fixed-point number type, defined as an alias of `int`.
- In `struct thread`
    - `nice` : The nice value of the thread.
    - `recent_cpu` : The recent CPU of the thread.

#### ALGORITHMS 

> C2: Suppose threads A, B, and C have nice values 0, 1, and 2.  Each
> has a recent_cpu value of 0.  Fill in the table below showing the
> scheduling decision and the priority and recent_cpu values for each
> thread after each given number of timer ticks:

timer  recent_cpu    priority   thread
ticks   A   B   C   A   B   C   to run
-----  --  --  --  --  --  --   ------
 0      0   0   0   63  61  59   A 
 4      4   0   0   62  61  59   A 
 8      8   0   0   61  61  59   B 
12      8   4   0   61  60  59   A 
16      12  4   0   60  60  59   B 
20      12  8   0   60  59  59   A 
24      16  8   0   59  59  59   C 
28      16  8   4   59  59  58   B 
32      16  12  4   59  58  58   A 
36      20  12  4   58  58  58   B 

> C3: Did any ambiguities in the scheduler specification make values
> in the table uncertain?  If so, what rule did you use to resolve
> them?  Does this match the behavior of your scheduler?

Yes. The rule is that the thread with the highest priority will be scheduled first. However, if there are multiple threads with the highest priority, the behavior is undefined. 

In the table above, the behavior is to use the first thread with the highest priority in the ready list, assuming that the initial order is A, B, C and the ready list acts in a FIFO manner. This may match the behavior of our scheduler, as it is exactly how `list_max` works.

> C4: How is the way you divided the cost of scheduling between code
> inside and outside interrupt context likely to affect performance?

The performance may be affected severely since the timer interrupt handler inside interrupt context will iterate all threads to update priority every 4 ticks.

#### RATIONALE 

> C5: Briefly critique your design, pointing out advantages and
> disadvantages in your design choices.  If you were to have extra
> time to work on this part of the project, how might you choose to
> refine or improve your design?

Our design is simple and trivia, which is an advantage for us to implement it. However, it is not efficient enough, as it will iterate all threads to update priority.

> C6: The assignment explains arithmetic for fixed-point math in
> detail, but it leaves it open to you to implement it.  Why did you
> decide to implement it the way you did?  If you created an
> abstraction layer for fixed-point math, that is, an abstract data
> type and/or a set of functions or macros to manipulate fixed-point
> numbers, why did you do so?  If not, why not?

Our implementation defines fixed-point number type `f32` as an alias of `int`. The reason is that it can help the type system and LSP to better check the type correctness in the code and avoid potential bugs.

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
