/// Decimal strings preserve protocol nanoseconds and counters across runtimes.
BigInt wireCounter(Object? value) {
  if (value is! String || !RegExp(r'^(0|[1-9][0-9]{0,18})$').hasMatch(value)) {
    throw const FormatException('Invalid decimal counter');
  }
  return BigInt.parse(value);
}
