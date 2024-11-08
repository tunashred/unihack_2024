import { Component } from '@angular/core';
import { FileUploaderComponent } from "../../components/file-uploader/file-uploader.component";


@Component({
  selector: 'app-home-page',
  standalone: true,
  templateUrl: './home-page.component.html',
  styles: ``,
  imports: [FileUploaderComponent]
})
export class HomePageComponent {

}
