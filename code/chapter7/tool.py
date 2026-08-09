class ToolRegisty:
 def __init__(self):
    self.tools = []

 def add_tool(self, tool):
    self.tools.append(tool)

 def get_tools(self):
    return self.tools

 def remove_tool(self, tool):
    self.tools.remove(tool)

 def get_tool_by_name(self, name):