"""Data presentation in a human-readable format (a table, html page, ..)"""

import copy


class Document:
    """Presenting data in a form of a document"""

    def __init__(self, preprocessors=None):
        """Preprocessors dict {type, [function(raw_data), ..]}"""
        self.preprocessors = preprocessors or []
        self.renderers = {"rows": self.render_rows,
                          "blocks": self.render_blocks,
                          "blockgroup": self.render_blockgroup,
                          "dl": self.render_dl,
                          "div": self.render_div,
                          "list": self.render_list,
                          "image": self.render_image,
                          "line": self.render_line,
                          "link": self.render_link,
                          "table": self.render_table}

    @staticmethod
    def _styles(styles):
        """Unwraps styles given by a dictionary"""
        return " ".join(map(lambda x: "%s: %s;" % x, styles.items()))

    def render_snippet(self, data, depth): # pylint: disable=too-many-branches
        """Generates recursively pieces of html"""
        if isinstance(data, list):
            acc = []
            for result in map(lambda x: self.render_snippet(x, depth), data):
                if isinstance(result, list):
                    acc += result
                else:
                    acc.append(result)
            return acc
        elif isinstance(data, str):
            return data
        elif isinstance(data, int):
            return str(data)
        elif isinstance(data, float):
            return "%.2f" % data

        assert isinstance(data, dict) and "type" in data, type(data)
        original_data = copy.deepcopy(data)
        if data["type"] in self.preprocessors:
            preprocessors = self.preprocessors[data["type"]]
            if not isinstance(preprocessors, list):
                preprocessors = [preprocessors]
            for preprocessor in preprocessors:
                data = preprocessor(data)

        if "depth" in data:
            depth = data["depth"]
        result = ""
        if data.get("anchor", None):
            result += "<a id='%s'></a>" % data['anchor']
        if "name" in data:
            result += "<h%s>%s</h%s>\n" % (depth, data["name"], depth)
            depth += 1
        if data["type"] in self.renderers:
            temp = self.renderers[data["type"]](data, depth)
            if isinstance(temp, list):
                result = temp
            else:
                result += temp
        else:
            raise ValueError("Unsupported type %s" % data["type"])

        if "link" in data:
            result = "<a href='%s'>%s</a>" % (data["link"] if data["link"] != True \
                    else original_data["source"], result)
        return result


    def render_rows(self, data, depth):
        """Render data corresponding to rows"""
        assert data["type"] == "rows"
        buf = ""
        if data.get("params", None):
            buf += self.render_snippet({"type": "dl", "data": data["params"]}, depth)
        for element in self.render_snippet(data["data"], depth):
            buf += "<div>\n%s\n</div>\n" % element
        return buf


    def render_blocks(self, data, depth):
        """Render data corresponding to blocks"""
        assert data["type"] == "blocks"
        child_styles = data["css"]+"; " if "css" in data else ""
        child_styles += "border: %spx solid %s; " % (data.get("line_width", 0),
                                                     data.get("line", "transparent"))
        buf = ""
        if data.get("params", None):
            buf += self.render_snippet({"type": "dl", "data": data["params"]}, depth)
        buf += "<div style='display: flex; flex-wrap: wrap;'>\n"
        for element in self.render_snippet(data["data"], depth):
            if "<div" in element.split(">", 1)[0]:
                buf += element
            else:
                buf += "<div style='%s'>%s</div>\n" % (child_styles, element)
        if data.get("fill", False):
            for _ in range(10):
                buf += "<div style='flex: auto;'></div>"
        buf += "</div>\n"
        return buf


    def render_blockgroup(self, data, depth):
        """Render data corresponding to a blockgroup"""
        assert data["type"] == "blockgroup"
        acc = []
        raw = self.render_snippet(data["data"], depth)
        for i, rawi in enumerate(raw):
            child_styles = data["css"]+"; " if "css" in data else ""
            border_style = "%spx solid %s" % (data.get("line_width", 2),
                                              data.get("line", "transparent"))
            child_styles += "border-top: %s; border-bottom: %s;" % (border_style, border_style)
            if i == 0:
                child_styles += " border-left: %s;" % border_style
            if i == len(raw)-1:
                child_styles += " border-right: %s;" % border_style
            acc.append("<div style='%s'>%s</div>" % (child_styles, rawi))
        return acc


    def render_dl(self, data, depth):
        """Render data corresponding to a dl"""
        assert data["type"] == "dl"
        buf = ""
        for piece in data["data"]:
            buf += "<dt>%s</dt>" % self.render_snippet(piece[0], depth)
            piece1 = piece[1] if isinstance(piece[1], (list, tuple)) else [piece[1]]
            for piece1i in piece1:
                buf += "<dd>%s</dd>\n" % self.render_snippet(piece1i, depth)
        buf = "<dl style='%s'>%s</dl>" % (data.get("css", ""), buf)
        return buf


    def render_div(self, data, depth):
        """Render data corresponding to a div"""
        assert data["type"] == "div"
        child = self.render_snippet(data["data"], depth) if "data" in data else ""
        buf = "<div style='%s'>%s</div>" % (data.get("css", ""), child)
        return buf


    def render_list(self, data, depth):
        """Render data corresponding to a list"""
        assert data["type"] == "list"
        buf = "".join(map(lambda x: "<li>%s</li>\n" % \
                self.render_snippet(x, depth), data["data"]))
        buf = ("<ol>%s</ol>" if data.get("ordered", False) else "<ul>%s</ul>") % buf
        return buf


    def render_line(self, data, depth):
        """Render data corresponding to a single line"""
        assert data["type"] == "line"
        return " | ".join(map(lambda x: self.render_snippet(x, depth), data["data"]))


    def render_image(self, data, depth):
        """Render data corresponding to an image"""
        assert data["type"] == "image"
        size_limit = "max-width: {size}; max-height: {size};".format(size=data["size"]) \
                if "size" in data else ""
        # to center: margin-left: auto; margin-right: auto; display: block;
        additional = " title='%s'" % data["title"].replace("'", "&#39;") if "title" in data else ""
        style = self._styles(data.get("style", {})) + data.get("css", "")
        result = "<figure style='margin: 0;%s'>" % style + \
                "<img{add} src='{source}' style='{size_limit}'></img>".format(source=data["source"],
                                                                              size_limit=size_limit,
                                                                              add=additional)
        if data.get("params", None):
            figcaption = data["params"]
            if isinstance(data["params"], dict):
                figcaption = self.render_snippet({"type": "dl", "data": figcaption}, depth)
            result += "<figcaption>" + figcaption + "</figcaption>"
        result += "</figure>"
        return result

    def render_table(self, data, depth):
        """Render data corresponding to a table"""
        assert data["type"] == "table"
        buf = ""
        if data.get("params", None):
            buf += self.render_snippet({"type": "dl", "data": data["params"]}, depth)
        if "header" in data:
            style = (' style="%s"' % data["header_style"]) if "header_style" in data else ""
            cell_str = "<th%s>%%s</th>" % style
            header = "".join(cell_str % self.render_snippet(x, depth) for x in data["header"])
            buf += '<tr class="header_row">%s</tr>\n' % header
        for i, row in enumerate(data["data"]):
            style = data.get("cell_style", "white-space: nowrap;")
            if "padding" in data:
                style += " padding: %s;" % data["padding"]

            buf += '<tr class="data_row row_%s">' % i
            for j, cell in enumerate(row):
                buf += '<td class="cell_%s" style="%s">%s</td>' % \
                        (j, style, self.render_snippet(cell, depth))
            buf += "</tr>\n"
        return "<table>%s</table>" % buf


    @staticmethod
    def render_link(data, _):
        """Render data corresponding to an image"""
        assert data["type"] == "link"
        return "<a href='%s'>%s</a>" % (data["source"], data.get("title", data["source"]))


    def struct2html(self, structure, output_file=None, css=None, script=None):
        """Converts data structure to an html. Save it to output_file if provided."""
        head = ""
        if css is not None:
            head += "<style>\n%s\n</style>" % css
        result = """
        <head>
        <meta charset='utf-8' />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        %s
        </head>
        <body>\n%s\n""" % (head, self.render_snippet(structure, 1))
        if script is not None:
            result += "\n%s\n" % script
        result += "</body>"
        if output_file:
            with open(output_file, 'w') as handler:
                handler.write(result)
        return result
