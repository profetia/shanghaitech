export function toValidLength(value) {
  if (value === null || value === undefined) {
    return 'auto';
  }
  if (typeof value === 'number' || !isNaN(new Number(value))) {
    return `${value}px`;
  }
  return value;
}
