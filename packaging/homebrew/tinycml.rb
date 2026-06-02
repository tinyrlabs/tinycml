class Tinycml < Formula
  desc "Tiny C Machine Learning Library — a lightweight C11 ML library with no dependencies"
  homepage "https://github.com/username/tinycml"
  # TODO: Replace with actual GitHub release URL and SHA256
  url "https://github.com/username/tinycml/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"

  depends_on "make" => :build

  def install
    system "make", "build"
    system "make", "install", "PREFIX=#{prefix}"
  end

  test do
    (testpath/"test.c").write <<~C
      #include <tinycml/tinycml.h>
      #include <stdio.h>

      int main(void) {
          TinyCMLMatrix *m = tinycml_matrix_create(2, 2);
          tinycml_matrix_set(m, 0, 0, 1.0);
          tinycml_matrix_set(m, 0, 1, 2.0);
          tinycml_matrix_set(m, 1, 0, 3.0);
          tinycml_matrix_set(m, 1, 1, 4.0);
          printf("%.1f\\n", tinycml_matrix_get(m, 0, 0));
          tinycml_matrix_free(m);
          return 0;
      }
    C
    system ENV.cc, "test.c", "-I#{include}", "-L#{lib}", "-ltinycml", "-lm", "-o", "test"
    assert_equal "1.0\n", shell_output("./test")
  end
end
