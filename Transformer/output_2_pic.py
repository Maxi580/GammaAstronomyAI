import matplotlib.pyplot as plt


def analyze_binary_file(filename="output.txt", chunk_size=1000):
    """
    Analyzes a binary file (consisting of 1s and 0s) and creates a diagram
    showing the distribution of 1s per chunk of lines.

    Args:
        filename: The path to the binary file.
        chunk_size: The number of lines to process in each chunk.
    """

    line_counts = []
    one_percentages = []

    with open(filename, "r") as f:
        chunk_count = 0
        ones_in_chunk = 0
        total_in_chunk = 0

        for line_number, line in enumerate(f):
            line_counts.append(line_number)
            ones_in_chunk += line.strip().count("1")
            total_in_chunk += len(line.strip())

            if (line_number + 1) % chunk_size == 0:
                percentage = (ones_in_chunk / total_in_chunk) * 100 if total_in_chunk > 0 else 0
                one_percentages.append(percentage)

                chunk_count += 1
                ones_in_chunk = 0
                total_in_chunk = 0

        # Handle any remaining lines in the last chunk
        if total_in_chunk > 0:
            percentage = (ones_in_chunk / total_in_chunk) * 100
            one_percentages.append(percentage)

    # Generate x-axis values based on chunk number or line number
    x_values = list(range(0, len(line_counts), chunk_size))  # Chunk-based x-axis

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, one_percentages)
    plt.xlabel("Line Number (in chunks of {})".format(chunk_size))
    plt.ylabel("Percentage of 1s")
    plt.title("Distribution of 1s in {}".format(filename))
    plt.grid(True)
    plt.show()


# Example usage:
analyze_binary_file("output.txt", chunk_size=1000)