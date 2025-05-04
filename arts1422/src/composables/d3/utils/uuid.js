let uid = 0;

export function getUUID() {
  uid += 1;
  return uid;
}
