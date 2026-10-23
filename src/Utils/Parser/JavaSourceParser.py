import javalang as jl
from .SourceRefiner import clear_formatting

class JavaSourceParser:
    def __init__(self, data, clear_formatting=False):
        self.data = data
        self.tree = jl.parse.parse(data)
        self.methods = {}
        self.fields = set()
        self.clear_formatting = clear_formatting

    def get_start_end_for_node(self, node_to_find):
        start = None
        end = None
        for path, node in self.tree:
            if start is not None and node_to_find not in path:
                end = node.position
                return start, end
            if start is None and node == node_to_find:
                start = node.position
        return start, end

    def get_string(self, start, end):
        if start is None:
            return ""

        end_pos = None
        if end is not None:
            end_pos = end.line - 1

        lines = self.data.splitlines(True)
        string = "".join(lines[start.line:end_pos])
        string = lines[start.line - 1] + string

        if end is None:
            left = string.count("{")
            right = string.count("}")
            if right - left == 1:
                p = string.rfind("}")
                string = string[:p]

        return string

    def parse_methods(self):
        for _, node in self.tree.filter(jl.parser.tree.MethodDeclaration):
            start, end = self.get_start_end_for_node(node)
            method_body = self.get_string(start, end)
            if self.clear_formatting:
                method_body = clear_formatting(method_body)
            self.methods[node.name] = method_body

        return self.methods

    def parse_fields(self):
        for _, node in self.tree.filter(jl.parser.tree.FieldDeclaration):
            start, end = self.get_start_end_for_node(node)
            self.fields.add(self.get_string(start, end).strip())
            # self.fields[node.name] = self.get_string(start, end)

        return self.fields

if __name__ == '__main__':
    data = '''public class Main {
    public int a;
        public static void main(String[] args) {
        int b;
            System.out.println("hello world");
        }

        public void foo() {
            System.out.println("foo");
        }

        public void bar() {
            System.out.println("bar");
        }
    }'''

    parser = JavaSourceParser(data, clear_formatting=True)
    parser.parse_methods()
    print(parser.methods)
    print(parser.parse_fields())
