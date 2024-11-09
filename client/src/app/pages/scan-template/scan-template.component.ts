import { Component } from '@angular/core';
import { FileUploaderComponent } from "../../components/file-uploader/file-uploader.component";

@Component({
  selector: 'app-scan-template',
  standalone: true,
  imports: [FileUploaderComponent],
  templateUrl: './scan-template.component.html',
})
export class ScanTemplateComponent {

}
