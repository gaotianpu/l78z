; "use strict";

//各种配置参数
const HIGHTLIGHT_COLOR = 'rgb(60,160,250)'; //高亮颜色
const DASHES_COLOR = 'rgb(60,60,60)'; //虚线颜色
const NORMAL_COLOR = 'rgb(30,30,30)'; //正常的颜色
const DUP_POINT_DISTANCE_THRESHOLD = 10; //磁吸距离

////////////////工具箱，枚举值
var Tools = {
    MOVE: 0, //0-移动
    POINT: 1, //点
    LINE: 2, //直线
    CIRCLE: 3, //圆 
    PERPENDICULAR_BISECTOR: 4, //中垂线
    ANGULAR_BISECTOR: 5, //角平分线  
    VERTICAL_LINE_THROUGH_POINT: 6, //过点垂线  
    PARALLEL_LINE: 7, //平行线
    COMPASSES: 8 //圆规 
};
if (Object.freeze) {
    Object.freeze(Tools);
}

///////////////////定义对象, 需要抽象一些方法到基类里

//点
class Point {
    constructor(x, y, text = false) {
        this.x = x;
        this.y = y;
        this.text = text;
    }

    get type() {
        return Tools.POINT;
    }

    get points() {
        return [this];
    }

    get first() {
        return this;
    }

    get last() {
        return this;
    }

    update_last_point(x, y) {
        // this.x = x;
        // this.y = y;
    }

    serialize() {
        return "{x:,y:,text}";
    }

    //与另外一点的距离
    distance(point) {
        var d = Math.hypot(this.x - point.x, this.y - point.y);
        return d;
    }

    //是否在canvas边界内？
    in_boundary(canvas_rect) {
        if (this.x < 0 || this.y < 0 || x > canvas_rect.width || y > canvas_rect.height) {
            //TODO: 画布平移、缩放？
            return false;
        }
        return true;
    }

    is_ok() {
        return true;
    }

    //计算给定一点与所有相交点之间的距离
    get_nearest_point(points) {
        var dist_list = []
        for (var item of points) {
            // var d = Math.sqrt((x - item.x) * (x - item.x) + (y - item.y) * (y - item.y));
            // console.log(point.type,item.type); 
            dist_list.push(this.distance(item));
        }

        var min_d = Math.min.apply(null, dist_list);
        if (min_d > DUP_POINT_DISTANCE_THRESHOLD) {
            return false;
        }

        for (var i = 0; i < dist_list.length; i++) {
            if (dist_list[i] == min_d) {
                return points[i];
            }
        }

        return false;
    }

    //绘制操作是否应该放置在这里？ 不应该放在这里model与view分离的原则
    // draw(ctx) {
    // } 
}

//线
class Line {
    constructor(start) {
        this.points = [];
        this.points.push(start);
        this.points.push(new Point(start.x, start.y));
    }

    get type() {
        return Tools.LINE;
    }

    update_last_point(x, y) {
        this.last.x = x;
        this.last.y = y;
    }

    get first() {
        return this.points[0];
    }

    get last() {
        return this.points[this.points.length - 1];
    }

    get x_diff() {
        return this.points[0].x - this.points[1].x;
    }

    get y_diff() {
        return this.points[0].y - this.points[1].y;
    }

    distance(point) {
        //线外一点，到直线距离
        //highlight 用于判定鼠标点是否在线上，如果接近，磁吸到线上最近的点

    }

    //线段长度
    get segment_length() {
        return this.points[0].distance(this.points[1]);
    }

    get is_ok() {
        var d = this.segment_length;
        if (d > 10) {
            return true;
        }
        return false;
    }

    //直线的斜率
    get weight() {
        return this.y_diff / this.x_diff;
    }

    //直线的偏置项
    get bias() {
        // y = wx + b -> b = y-wx
        return this.points[0].y - this.weight * this.points[0].x;
    }

    //线段的中心点
    get center() {

    }

    //边界点是直线的属性吗？ 需要引入外部依赖？

    //左边界点
    get left() {
        return new Point(0, this.bias);
    }

    //右边界点

    right_point(max_x) {
        return new Point(max_x, this.weight * max_x + this.bias);
    }

    //与另外一条直线的交点
    line_intersect_point(line) {
        //平行或重合
        if (this.weight == line.weight) {
            return false;
        }

        var x = (this.bias - line.bias) / (line.weight - this.weight);
        var y = this.weight * x + this.bias;

        return new Point(x, y);
    }

    //与圆的交点

    //给定某点是否在这条直线上,线段里？
    // distance(point){ 
    // }

    //绘制方式，是否高亮等，非直线的固有属性？ 
}

//圆
class Circle {
    constructor(circle_center) {
        this.points = []
        this.points.push(circle_center);
        this.points.push(new Point(circle_center.x, circle_center.y));
    }

    get type() {
        return Tools.CIRCLE;
    }

    get first() {
        return this.points[0];
    }

    get last() {
        return this.points[this.points.length - 1];
    }

    update_last_point(x, y) {
        this.last.x = x;
        this.last.y = y;
    }

    get x() {
        return this.points[0].x;
    }

    get y() {
        return this.points[0].y;
    }

    get center() {
        return this.points[0];
    }

    distance(point) {
        //线外一点，到圆周上的距离

    }

    //半径
    get radius() {
        return this.center.distance(this.points[1]);
        // return Math.hypot(this.points[0].x - this.points[1].x, this.points[0].y - this.points[1].y);
    }

    get is_ok() {
        var d = this.radius;
        if (d > 10) {
            return true;
        }
        return false;
    }

    line_intersect_point(line) {
        // 直线方程： y = ax + b 
        // 圆方程： (x-x0)²+(y-y0)²=r²  //圆心坐标为(x0，y0),r半径 
        // 推导：
        // (x-x0)² + (ax+b-y0)²=r² //相交点x,y坐标相等，联立方程，求x值  
        // (x²+x0²-2*x0*x) + (a²x² + (b-y0)² + 2*a*x*(b-y0) )= r²
        // (x² + a²x²) + (2*a*(b-y0)*x-2*x0*x) + (x0² + (b-y0)²) = r²
        // (a²+1)x² + 2*(a*(b-y0)-x0)*x + (x0² + (b-y0)²) = r²
        // (a²+1)x² + 2*(a*(b-y0) - x0) * x = r² - (b-y0)² - x0²
        // x² + 2*((a*(b-y0) - x0)/(a²+1)) * x = (r² - (b-y0)² - x0²)/(a²+1)
        // x² + 2*((a*(b-y0) - x0)/(a²+1)) * x + ((a*(b-y0) - x0)/(a²+1))² = (r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²
        // (x+ [((a*(b-y0) - x0)/(a²+1))²]开方)² = (r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²
        // x+ [((a*(b-y0) - x0)/(a²+1))²]开方 = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放
        // x = [(r² - (b-y0)² - x0²)/(a²+1) + ((a*(b-y0) - x0)/(a²+1))²]开放 - [((a*(b-y0) - x0)/(a²+1))²]开放 
        var A = Math.pow(line.weight, 2) + 1;
        var B = line.bias - this.y;
        var C = (B * line.weight - this.x) / A
        var t = (Math.pow(this.radius, 2) - Math.pow(this.x, 2) - Math.pow(B, 2)) / (Math.pow(line.weight, 2) + 1)
        var f = t + Math.pow(C, 2);
        if (f < 0) {
            //不存在交点
            //工程上应该允许有个误差值？误差多少合适？相切的点？
            //过圆心、垂直与直线的线段，线段的距离是否等于半径？
            return false;
        }
        var l = Math.sqrt(f)

        var ret = [];

        var x1 = l - C;
        var y1 = line.weight * x1 + line.bias;
        // if (in_boundary(x1, y1)) {
        // ret.push(  { 'x': Math.round(x1), 'y': Math.round(y1) });
        ret.push(new Point(x1, y1));
        // }

        if (l != 0) {
            var x2 = - l - C;
            var y2 = line.weight * x2 + line.bias;
            // if (in_boundary(x2, y2)) {
            ret.push(new Point(x2, y2));

            // ret.push({ 'x': Math.round(x2), 'y': Math.round(y2) });
            // }
        }
        return ret;

    }

    //与另外一个圆的交点
    circle_intersect_point(circle) {
        //1. 两个圆心连线的长度，大于两圆半径和->无交点，等于半径和->1个交点，小于半径和->2个交点
        //2. 如果有2个交点，则交点之间的连线垂直于圆心之间的连线，可计算交点连线的斜率
        //3. 如果有1个交点
        // 如果两个圆有交点，则交点的连线必然垂直于，两个圆心连接的直线 
        // 圆方程： 
        // (x-x0)²+(y-y0)²=r0²
        // (x-x1)²+(y-y1)²=r1²
        // (x-x1)²+(y-y1)² - ( (x-x0)² + (y-y0)² ) = r1²-r0²
        // (x² + x1² - 2*x1*x + y²+y1²-2*y1*y) - (x² + x0² - 2*x0*x + y²+y0²-2*y0*y) = r1²-r0²
        // x1² - 2*x1*x + y1²-2*y1*y - x0² + 2*x0*x - y0² + 2*y0*y = r1²-r0²
        // 2*x0*x - 2*x1*x + 2*y0*y -2*y1*y = r1²-r0² + y0² + x0² - y1² - x1²
        // (x0-x1)*x + (y0-y1)*y =  (r1²-r0² + y0² + x0² - y1² - x1²)/2 
        // y + ((x0-x1)/(y0-y1))*x =  (r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1)
        // y = - ((x0-x1)/(y0-y1))*x + (r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1)

        // y = Ax + B ? A=((x1-x0)/(y0-y1)) B=(r1²-r0² + y0² + x0² - y1² - x1²)/2/(y0-y1) 

        // x² + x0² - 2*x0*x + (Ax+B -y0)² = r0² 
        // x² - 2*x0*x + A²*x² + 2*(B-y0)*A*x  = r0² - x0² - (B-y0)²
        // (A²+1)*x² + 2*((B*A-y0*A-x0))*x = r0² - x0² - (B-y0)²
        // x² + 2* ((B*A-y0*A-x0)/(A²+1) )*x = (r0² - x0² - (B-y0)²)/(A²+1) 
        // error: (A²+1)*x² + 2*(B-y0-x0)*x = r0² - x0² - (B-y0)² 
        // C = ((B*A-y0*A-x0)/(A²+1))
        // x² + 2*C*x + C² = (r0² - x0² - (B-y0)²)/(A²+1) + C² 

        var center_distance = this.center.distance(circle.center);
        var radius = this.radius + circle.radius;
        if (center_distance > radius) {
            //2圆心距离大于半径和
            return false;
        }

        var A = (this.x - circle.x) / (circle.y - this.y);
        //B= (r1²-r0² + x0² - x1² + y0² - y1²)/2/(y0-y1)
        var B0 = Math.pow(this.radius, 2) - Math.pow(circle.radius, 2) + Math.pow(circle.x, 2) - Math.pow(this.x, 2) + Math.pow(circle.y, 2) - Math.pow(this.y, 2);
        var B = B0 / 2 / (circle.y - this.y);

        //C = ((B*A-y0*A-x0)/(A²+1))
        var C = ((B - circle.y) * A - circle.x) / (Math.pow(A, 2) + 1);
        // d0 = (r0² - x0² - (B-y0)²)/(A²+1) 
        var d0 = (Math.pow(circle.radius, 2) - Math.pow(circle.x, 2) - Math.pow(B - circle.y, 2)) / (Math.pow(A, 2) + 1)
        var d1 = d0 + Math.pow(C, 2);
        if (d1 < 0) {
            return false;
        }

        var ret = [];
        var D = Math.sqrt(d1);

        var x = - D - C;
        var y = A * x + B
        // if (in_boundary(x, y)) {
        ret.push(new Point(x, y));

        // ret.push({ 'x': Math.round(x), 'y': Math.round(y) });
        // }

        if (d1 != 0) {
            var x1 = D - C;
            var y1 = A * x1 + B;
            // if (in_boundary(x1, y1)) {
            ret.push(new Point(x1, y1));

            // ret.push({ 'x': Math.round(x1), 'y': Math.round(y1) });
            // }
        }

        return ret;
    }
}

////////////// main 主函数 ////
; (function () {
    function main() {

        //页面元素
        //交互按钮
        var toolbox = document.getElementById("toolbox");
        var btn_undo = document.getElementById("btn_undo");
        var btn_redo = document.getElementById("btn_redo");
        // 画布
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        var canvas_rect = canvas.getBoundingClientRect();

        //定义各种变量 
        var current_tool = Tools.MOVE; //当前用户选择的绘图工具，默认移动  
        var isDrawing = false;
        var requestAnimationFrame_status = 0;
        var movie_points = { 'x': 0, 'y': 0, 'x_movie': 0, 'y_movie': 0 };

        var operation_list = []; //用户的绘图记录
        var undo_operation_list = []; //撤销操作临时保存   


        //过圆心且垂直与线段的直线
        function calc_chuizhidian(line_item, circle) {
            // return false;
            // //TODO:
            // //过圆心且垂直与线段的直线，与
            // //计算直线上、距离圆心最近的点, 可以用来判断圆的切线
            // //设该点为x,y
            // //圆心-线段端点 距离
            // // a² + b² = c²
            // // c² = 
            // var line = calc_line_parameters(line_item);
            // var radius = calc_distance(circle);

            // var cc = Math.pow(circle.x - line_item.x, 2) + Math.pow(circle.y - line_item.y, 2);
            // var A = 2 * (Math.pow(line.weight, 2) + 1);
            // var B = (2 * line.bias - circle.x - circle.y - line_item.x - line_item.y) / A;
            // var C = Math.pow(circle.x, 2) + Math.pow(line.bias - circle.y, 2) + Math.pow(line_item.x, 2) + Math.pow(line.bias - line_item.y, 2);

            // var D = (cc - C) / A;
            // var E = (cc - C) / A + Math.pow(B, 2);



            // if (E < 0) {
            //     //不存在交点
            //     //工程上应该允许有个误差值？误差多少合适？相切的点？
            //     // 过圆心、垂直与直线的线段，线段的距离是否等于半径？
            //     return false;
            // }
            // var l = Math.sqrt(E)

            // var x1 = l - B
            // var y1 = line.weight * x1 + line.bias;

            // var x2 = -l - B
            // var y2 = line.weight * x1 + line.bias;

            // var ret = [{ 'x': Math.round(x1), 'y': Math.round(y1) },
            // { 'x': Math.round(x2), 'y': Math.round(y2) }];

            // return ret;

        }

        //人工绘制的点
        function get_manual_points(include_last = true) {
            var tmp = [];
            var len_operation_list = operation_list.length;
            if (!include_last) {
                len_operation_list = len_operation_list - 1;
            }

            for (var i = 0; i < len_operation_list; i++) {
                tmp.push.apply(tmp, operation_list[i].points);
            }

            //去重？
            var manual_points = [];
            for (var i = 0; i < tmp.length; i++) {
                var d = tmp[i].get_nearest_point(manual_points);
                if (!d) {
                    manual_points.push(tmp[i]);
                }
            }

            return manual_points;
        }

        //获取所有的相交点
        function get_intersect_points(include_last = true) {
            var tmp_points = [];

            //计算两个图形相交的点
            var len = operation_list.length;
            if (len < 2) {
                return tmp_points;
            }
            if (!include_last) {
                len = len - 1;
            }
            for (var i = 0; i < len; i++) {
                for (var j = i + 1; j < len; j++) {
                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.LINE) {
                        //两直线相交
                        var point = operation_list[i].line_intersect_point(operation_list[j]);
                        if (point) {
                            tmp_points.push(point);
                        }
                    }

                    if (operation_list[i].type == Tools.LINE && operation_list[j].type == Tools.CIRCLE) {
                        //直线&圆 
                        var points = operation_list[j].line_intersect_point(operation_list[i]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }

                    }

                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.LINE) {
                        //圆&直线 
                        var points = operation_list[i].line_intersect_point(operation_list[j]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }

                    }
                    if (operation_list[i].type == Tools.CIRCLE && operation_list[j].type == Tools.CIRCLE) {
                        //圆&圆 
                        var points = operation_list[i].circle_intersect_point(operation_list[j]);
                        if (points) {
                            tmp_points.push.apply(tmp_points, points);
                        }
                    }
                }

            }

            //用户绘制点+相交点去重
            var intersect_points = [];
            var manual_points = get_manual_points(false);
            for (var i = 0; i < tmp_points.length; i++) {
                var dm = tmp_points[i].get_nearest_point(manual_points);
                var d = tmp_points[i].get_nearest_point(intersect_points);
                if (!d && !dm) {
                    intersect_points.push(tmp_points[i]);
                }
            }
            return intersect_points;
        }



        function has_same_objects(last_item) {
            //判断是否存在重复的绘图对象 
            var len_operation_list = operation_list.length;
            for (var i = 0; i < len_operation_list; i++) {
                var item = operation_list[i];

                //2条线(w,b的差值)，
                if (last_item.type == Tools.LINE) {
                    if (Math.abs(last_item.weight - item.weight) < 0.01 ||
                        Math.abs(last_item.bias - item.bias) < 2) {
                        return true;
                    }
                }

                //2个圆(radius)
                if (last_item.type == Tools.CIRCLE) {
                    if (Math.abs(last_item.radius - item.radius) < 1) {
                        return true;
                    }
                }
            }

            return false;
        }

        //数据处理，类似mvc里的controler
        function process_data() {
            var len_operation_list = operation_list.length;
            if (len_operation_list < 1) {
                return false;
            }

            var last_index = len_operation_list - 1;
            var last_item = operation_list[last_index];
            var last_item_point = last_item.last;

            for (var item of operation_list) {
                //移动画布
                if (current_tool == Tools.MOVE) {
                    for (var point of item.points) {
                        point.x = point.x + movie_points.x_movie;
                        point.y = point.y + movie_points.y_movie;
                    }
                }

                //高亮
                item.highlight = false;
                if (isDrawing && current_tool != Tools.MOVE) {
                    if (item.type == Tools.LINE) {
                        // y = ax+b
                        var y = item.weight * last_item_point.x + item.bias;
                        if (Math.abs(y - last_item_point.y) < 2) {
                            item.highlight = true;
                        }
                    }

                    if (item.type == Tools.CIRCLE) {
                        // (x-x0)²+(y-y0)²=r² 
                        var r = Math.sqrt(Math.pow(last_item_point.x - item.x, 2) + Math.pow(last_item_point.y - item.y, 2));
                        if (Math.abs(r - item.radius) < 2) {
                            item.highlight = true;
                        }
                    }
                }
            }

            //线重合、圆重合？

            //磁吸到线？
            //线外点，距离线上最近点的距离小于某个值

            //磁吸到点
            if (isDrawing && current_tool != Tools.MOVE) {
                var points = get_manual_points(false);
                points.push.apply(points, get_intersect_points(false));

                // 判断起点
                var p0 = last_item.first.get_nearest_point(points);
                if (p0) {
                    last_item.points[0].x = p0.x;
                    last_item.points[0].y = p0.y;
                }

                // 判断终点
                if (last_item.type == Tools.LINE || last_item.type == Tools.CIRCLE) {
                    var p1 = last_item.last.get_nearest_point(points);
                    if (p1) {
                        last_item.update_last_point(p1.x, p1.y);
                    }
                }
            }


        }


        /// VIEW RENDER
        //画点
        function draw_point(point, strokeStyle, lineWidth) {
            ctx.beginPath();
            if (point.highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
            }
            ctx.arc(point.x, point.y, lineWidth, 0, Math.PI * 2, true);
            ctx.closePath();
            ctx.stroke();

            //如果Point需要标记文本 
            if (point.text) {
                ctx.font = "15px Verdana";
                ctx.fillText(point.text, point.x + 3, point.y);
            }

        }

        //绘制直线
        function draw_line(start, end, highlight, strokeStyle, lineWidth) {
            ctx.beginPath();
            ctx.lineWidth = lineWidth;
            if (highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
                //锯齿现象比较明显
            }
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.closePath();
            ctx.stroke();

        }

        //绘制圆
        function draw_arc(circle, strokeStyle) {
            ctx.beginPath();
            if (circle.highlight) {
                ctx.strokeStyle = HIGHTLIGHT_COLOR;
            } else {
                ctx.strokeStyle = strokeStyle;
            }
            ctx.arc(circle.x, circle.y, circle.radius, 0, Math.PI * 2);
            ctx.closePath();
            ctx.stroke();
        }

        function render() {
            ctx.clearRect(0, 0, canvas_rect.width, canvas_rect.height); //TODO 

            //根据操作记录和撤销记录设置undo/redo按钮是否可用   

            //从render中移出，有变化时再执行？
            process_data();

            //绘制人工点
            var manual_points = get_manual_points();
            for (var item of manual_points) {
                draw_point(item, NORMAL_COLOR, 2);
            }

            //绘制交点
            var intersect_points = get_intersect_points();
            for (var item of intersect_points) {
                draw_point(item, DASHES_COLOR, 1.5);
            }

            for (var item of operation_list) {
                //线 
                if (item.type == Tools.LINE) {
                    //绘制线段外虚直线部分 
                    draw_line(item.left, item.right_point(canvas_rect.width), item.highlight, DASHES_COLOR, 0.2);
                    //绘制线段
                    draw_line(item.first, item.last, item.highlight, NORMAL_COLOR, 1);
                }

                //圆
                if (item.type == Tools.CIRCLE) {
                    draw_arc(item, NORMAL_COLOR);
                }
            }

        }

        //事件绑定
        function event_binding() {
            //绘图工具框 
            toolbox.addEventListener('click', function () {
                var e = event || window.event;
                if (e.target && e.target.nodeName.toUpperCase() == "INPUT") {
                    current_tool = e.target.value;
                }
            }, false);

            //undo
            btn_undo.addEventListener('click', function () {
                var last = operation_list.pop();
                if (last) {
                    undo_operation_list.push(last);
                    render();
                }
            });

            //redo 
            btn_redo.addEventListener('click', function () {
                var last = undo_operation_list.pop();
                if (last) {
                    operation_list.push(last);
                    render();
                }
            });

            //两种习惯：暂时先支持第一种
            //1. 拖拽式 
            //2. 点击式 1.鼠标点击2下确定一条直线，一个圆等  
            canvas.addEventListener('mousedown', function (e) {
                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                isDrawing = true;

                if (current_tool == Tools.MOVE) {
                    movie_points.x = x;
                    movie_points.y = y;
                } else {
                    var point = new Point(x, y);

                    var obj = point;
                    if (current_tool == Tools.LINE) {
                        obj = new Line(point);
                    }
                    if (current_tool == Tools.CIRCLE) {
                        obj = new Circle(point);
                    }
                    operation_list.push(obj);
                }
                render();
            });
            canvas.addEventListener('mousemove', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0 || !isDrawing) {
                    return false;
                }

                var x = e.clientX - canvas_rect.left;
                var y = e.clientY - canvas_rect.top;

                if (current_tool == Tools.MOVE) {
                    //移动画布
                    movie_points.x_movie = x - movie_points.x;
                    movie_points.y_movie = y - movie_points.y;
                    movie_points.x = x;
                    movie_points.y = y;
                } else {
                    operation_list[last_index].update_last_point(x, y);
                }

                requestAnimationFrame_status = window.requestAnimationFrame(render);

            });
            canvas.addEventListener('mouseup', function (e) {
                var last_index = operation_list.length - 1;
                if (last_index < 0 || !isDrawing) {
                    return false;
                }

                //清空undo记录
                undo_operation_list = [];

                //起止距离太短等，非有效绘图
                if (!operation_list[last_index].is_ok) {
                    operation_list.pop();
                }

                isDrawing = false;
                window.cancelAnimationFrame(requestAnimationFrame_status);
                render();
            });

            canvas.addEventListener('click', function (e) {
                //
            });
            //右键单击、取消？

            //浏览器大小调整
            window.addEventListener('resize', function (e) {
                //设置canvas大小 
                canvas.height = document.documentElement.clientHeight;
                canvas.width = document.documentElement.clientWidth;
                canvas_rect = canvas.getBoundingClientRect();
                requestAnimationFrame_status = window.requestAnimationFrame(render);
            });

            window.addEventListener('load', function (e) {
                //页面加载，load用户的历史操作记录，呈现给用户？
            });
        }


        event_binding();
    }

    main();

})();