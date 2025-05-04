#ifndef VM_FRAME_H
#define VM_FRAME_H

#include <list.h>
#include "vm/page.h"

struct frame_table_entry {
    uint32_t *frame;
    struct thread *owner;
    struct sup_page_table_entry *page_entry;

    struct list_elem elem;
};

void frame_table_init (void);

struct frame_table_entry* frame_alloc (struct sup_page_table_entry *page_entry, 
    uint32_t* user_vaddr, bool writable);
void frame_free (struct frame_table_entry *fte);


#endif // VM_FRAME_H