# Communicating the purpose of software [TODO: Finish]

I work at Succinct, where we're building SP1, a provable compute engine. What does "provable" mean?

I can generate a [proof]() for the correct output of a {program, inputs} that can be verified for correctness quickly (faster than re-execution), without
necessarily revealing all of the inputs to the program.

This is useful in two contexts:

1. Environments with limited computational capacity: Blockchains, embedded devices, etc.
2. When inputs are sensitive: identity, financial data, etc.

## How do you build a provable compute engine?

We want to execute programs in a way that is general-purpose[INSERT HYPERLINK HERE], 